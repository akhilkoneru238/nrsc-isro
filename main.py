from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response
from pydantic import BaseModel
from typing import Optional, List, Literal, Union, Tuple

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont
import base64
import csv
import io
import time
import numpy as np
from inference import YOLODetector, RCNNDetector, ModelUnavailable, predict_multiscale
from inference import (DEFAULT_WEIGHTS1, DEFAULT_IOU, DEFAULT_WEIGHTS2, DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_MAX_DET,DEVICE,AVAILABLE_SCALES,AVAILABLE_SCALES_RCNN)
from schemas import Detection, DetectResponse, Health, ModelInfo, ModelName

VERSION = "2.0.0"
MAX_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 80_00_000
BOX_COLOR = (0, 255, 0)
BOX_WIDTH = 2

Image.MAX_IMAGE_PIXELS = MAX_PIXELS
app = FastAPI(title="Lunar Crater Detection using FastAPI",
              description="detecting craters on lunar surface using YOLO and faster R-CNN", version=VERSION)

app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_headers=["*"],
                   allow_methods=["*"])


async def __read__image(file: UploadFile) -> Tuple[np.ndarray, Image.Image]:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Uploaded images should be of the type .png, .jpg or .tiff")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "file is empty")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "The file added is too big.Upload a smaller file.Max 64 bytes")
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
        pil = pil.convert("RGB")
    except Image.DecompressionBombError:
        raise HTTPException(413, f"The image has too many pixels {MAX_PIXELS / 1_00_000}.Crop the image first")
    except Exception:
        raise HTTPException(413, "The image could not be loaded.Try again")

    if pil.width * pil.height > MAX_PIXELS:
        raise HTTPException(413,
                            f"The image is {pil.width * pil.height // 1e6:.0f} MP\n. Which exceeds the image width of {MAX_PIXELS // 1_00_000}\n. Crop it or run offline")

    return np.asarray(pil), pil


REGISTRY = {
    "yolo": YOLODetector(),
    "faster-rcnn": RCNNDetector()
}
def get_detector(model: ModelName):
    if model == "yolo":
        return REGISTRY["yolo"]
    elif model == "faster-rcnn":
        return REGISTRY["faster-rcnn"]
    else:
        raise KeyError(model)
def _run(
        model: str,
        arr: np.ndarray,
        iou: float,
        max_det: int,
        imgsz: int,
        conf: float,
        multiscale: bool = False,
        imgsz_list: Optional[List[int]] = None,
        merge_iou: float = 0.5, ) -> Tuple[np.ndarray, float]:
    try:
        detector = get_detector(model)
    except KeyError:
        raise HTTPException(400, f"Unknown model {model}. Use YOLO or R-CNN")

    t0 = time.perf_counter()

    try:
        # Multi-scale inference is now supported for both detectors:
        # predict_multiscale() only depends on detector.predict() accepting
        # (image, conf, imgsz, iou, max_det), which both YOLODetector and
        # RCNNDetector already implement — RCNNDetector.predict() swaps
        # model.transform.min_size/max_size per call and restores them
        # afterward, so it's safe to call in a loop across scales.
        if multiscale and model in ("yolo", "faster-rcnn"):
            if imgsz_list:
                scales = imgsz_list
            elif model == "faster-rcnn":
                # Faster R-CNN wasn't trained across a wide scale range like
                # YOLO — default to scales centered on its actual training
                # imgsz (640) rather than reusing YOLO's [10..1024] sweep.
                scales = AVAILABLE_SCALES_RCNN
            else:
                scales = [imgsz]
            boxes = predict_multiscale(detector, arr, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,merge_iou=merge_iou, scale_list=scales)
        else:
            boxes = detector.predict(arr, iou=iou, max_det=max_det, imgsz=imgsz, conf=conf)

    except ModelUnavailable as e:
        raise HTTPException(503, f"{detector.display_name} unavailable")

    except Exception as e:
        print("INFERENCE ERROR:", repr(e))
        raise HTTPException(400, f"Inferencing error: {str(e)}")

    return boxes, (time.perf_counter() - t0) * 1000.0

def _to_detection(boxes: np.ndarray) -> List[Detection]:
    if boxes.size == 0:
        return []
    order = boxes[:, 4].argsort()[::-1]

    out: List[Detection] = []
    for i, row in enumerate(boxes[order], start=1):
        x1, y1, x2, y2, score = (float(v) for v in row)
        width = x2 - x1
        height = y2 - y1
        out.append(Detection(
            id=i,
            x=round(x1, 2),
            y=round(y1, 2),
            w=round(width, 2),
            h=round(height, 2),
            cx=round((x1 + width / 2), 2),
            cy=round((y1 + height / 2), 2),
            diameter=round((width + height) / 2, 2),
            score=round(score, 4), ))

    return out


def _draw(pil: Image.Image, boxes: np.ndarray, labels: bool):
    img = pil.copy()
    draw = ImageDraw.Draw(img)
    lw = max(BOX_WIDTH, round(img.width / 700, 2))

    for i, row in enumerate(boxes, start=1):
        x1, y1, x2, y2, _ = (float(v) for v in row)
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=lw)
        if labels:
            draw.text((x1, max(0, y1 - 12)), f"#{i}", fill=BOX_COLOR)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _model_info(d) -> ModelInfo:
    available, detail = True, None
    if not d.loaded:
        try:
            d.ensure_loaded()
        except ModelUnavailable as e:
            available, detail = False, str(e)
        except Exception as e:
            available, detail = False, f"e__name__ : {str(e)}"

    return ModelInfo(name=d.name, display_name=d.display_name, loaded=d.loaded, weights=d.weights, available=available,
                     detail=detail, device=DEVICE)


@app.get("/api/health", response_model=Health)
def health():
    return Health(
        status="ok",
        version=VERSION,
        device=DEVICE,
        defaults={
            "conf": DEFAULT_CONF,
            "iou": DEFAULT_IOU,
            "max_det": DEFAULT_MAX_DET,
            "imgsz": DEFAULT_IMGSZ,

        },
        models=[_model_info(d) for d in REGISTRY.values()]
    )


@app.get("/api/health", response_model=List[ModelInfo])
def models():
    return [_model_info(d) for d in REGISTRY.values()]


@app.post("/api/detect", response_model=DetectResponse)
async def detect(
        image: UploadFile = File(..., description="Images on Lunar Surfaces"),
        model: ModelName = Form("yolo"),
        conf: float = Form(DEFAULT_CONF, ge=0.0, le=1.0),
        iou: float = Form(DEFAULT_IOU, ge=0.0, le=1.0),
        imgsz: int = Form(DEFAULT_IMGSZ, ge=10, le=4096),
        max_det: int = Form(DEFAULT_MAX_DET, ge=1, le=300000),
        multiscale: bool = Form(True),
        imgsz_list: Optional[str] = Form(None),
        merge_iou: float = Form(0.5, ge=0.0, le=1.0),
        render: bool = Form(False, description="Also return a base64 annotated PNG"),
        labels: bool = Form(False, description="Draw #id and score on the overlay"), ):
    arr, pil = await __read__image(image)
    scales = None
    if multiscale and imgsz_list:
        try:
            scales = [int(s.strip()) for s in imgsz_list.split(",") if s.strip()]
        except ValueError:
            raise HTTPException(400, "imgsz_list must be comma-separated integers, e.g. '640,1024,1600'")

    boxes, ms = _run(model, arr, iou, max_det, imgsz, conf, multiscale=multiscale, imgsz_list=scales,
                     merge_iou=merge_iou)
    dets = _to_detection(boxes)

    overlay = None
    if render:
        overlay = "data:image/png;base64," + base64.b64encode(_draw(pil, boxes, labels)).decode()

    return DetectResponse(model=model, image_width=pil.width, image_height=pil.height, conf=conf, iou=iou, imgsz=imgsz,
                          max_det=max_det, overlay_png=overlay, count=len(dets), detections=dets,
                          inference_ms=round(ms, 1))


@app.post("/api/detect/overlay")
async def detect_overlay(
        image: UploadFile = File(...),
        model: ModelName = Form("yolo"),
        iou: float = Form(DEFAULT_IOU, ge=0.0, le=1.0),
        conf: float = Form(DEFAULT_CONF, ge=0.0, le=1.0),
        max_det: int = Form(DEFAULT_MAX_DET, ge=1, le=300000),
        imgsz: int = Form(DEFAULT_IMGSZ, ge=10, le=4096),
        multiscale: bool = Form(True),
        imgsz_list: Optional[str] = Form(None),
        merge_iou: float = Form(0.5, ge=0.0, le=1.0),
        labels: bool = Form(False), ):
    arr, pil = await __read__image(image)
    scales = None
    if multiscale and imgsz_list:
        try:
            scales = [int(s.strip()) for s in imgsz_list.split(",") if s.strip()]
        except ValueError:
            raise HTTPException(400, "imgsz_list must be comma-separated integers, e.g. '640,1024,1600'")

    boxes, ms = _run(model, arr, iou, max_det, imgsz, conf, multiscale=multiscale, imgsz_list=scales,
                     merge_iou=merge_iou)
    return Response(
        content=_draw(pil, boxes, labels),
        media_type="image/png",
        headers={"x-crater-count": str(len(boxes)), "x-model": str(model)},
    )


@app.post("/api/detect/csv")
async def detect_csv(
        image: UploadFile = File(...),
        model: ModelName = Form("yolo"),
        iou: float = Form(DEFAULT_IOU, ge=0.0, le=1.0),
        multiscale: bool = Form(True),
        imgsz_list: Optional[str] = Form(None),
        merge_iou: float = Form(0.5, ge=0.0, le=1.0),
        conf: float = Form(DEFAULT_CONF, ge=0.0, le=1.0),
        max_det: int = Form(DEFAULT_MAX_DET, ge=1, le=300000),
        imgsz: int = Form(DEFAULT_IMGSZ, ge=10, le=4096),
):
    arr, pil = await __read__image(image)
    scales = None
    if multiscale and imgsz_list:
        try:
            scales = [int(s.strip()) for s in imgsz_list.split(",") if s.strip()]
        except ValueError:
            raise HTTPException(400, "imgsz_list must be comma-separated integers, e.g. '640,1024,1600'")

    boxes, ms = _run(model, arr, iou, max_det, imgsz, conf, multiscale=multiscale, imgsz_list=scales,
                     merge_iou=merge_iou)
    dets = _to_detection(boxes)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "x", "y", "w", "h", "cx", "cy", "diameter", "label"])
    for d in dets:
        writer.writerow([d.id, d.x, d.y, d.w, d.h, d.cx, d.cy, d.diameter,d.label])

    stem = (image.filename or "image").rsplit(".", 1)[0]
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}_detections.csv"'},
    )