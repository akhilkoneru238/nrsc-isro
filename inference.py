import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.transforms.functional import to_tensor
import numpy as np
from ultralytics import YOLO
import cv2
from pathlib import Path
from torchvision.ops import nms
from PIL import Image

AVAILABLE_SCALES = [10,120,240,480,640,1024]

# Faster R-CNN (best_frcnn.pt) was trained at a single fixed imgsz=640
# (confirmed via ckpt["args"]) — unlike YOLO, it was never trained across
# a wide range of scales. Reusing AVAILABLE_SCALES (tuned for YOLO, up to
# 1024) tests it at scales as small as 10px, which is meaningless for a
# model calibrated around 640px and dilutes the merged multi-scale result
# with low-confidence noise from mismatched scales.
AVAILABLE_SCALES_RCNN = [480, 640, 800]

DEFAULT_IOU = 0.6
DEFAULT_CONF = 0.3
DEFAULT_IMGSZ = 1024
DEFAULT_MAX_DET = 3000
DEVICE = "cpu"

basedir = Path(__file__).resolve().parent
DEFAULT_WEIGHTS1 = str(basedir / "models" / "best.pt")
DEFAULT_WEIGHTS2 = str(basedir / "models" / "best_frcnn.pt")


class ModelUnavailable(Exception):
    pass

class YOLODetector:
    def __init__(self,weights=DEFAULT_WEIGHTS1):

        self.weights = weights
        self.model = None

        self.name = "yolo"
        self.display_name = "YOLO"
        self.loaded = False

    def ensure_loaded(self):

        if self.model is None:
            self.model = YOLO(self.weights)

        self.loaded = True

        return self.model

    def load(self):
        return self.ensure_loaded()
    def predict(self,image,conf=DEFAULT_CONF,imgsz=DEFAULT_IMGSZ,iou=DEFAULT_IOU,max_det=DEFAULT_MAX_DET):
        self.load()

        print(f"YOLO settings -> conf={conf}, iou={iou}, max_det={max_det}")
        results = self.model.predict(source=image,conf=conf,imgsz=imgsz,iou=iou,max_det=max_det,verbose=False,device="cpu")
        print("YOLO raw detections:",len(results[0].boxes))

        boxes_output = []

        for result in results:
            for box in result.boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                score = box.conf[0].cpu().numpy()
                boxes_output.append([x1,y1,x2,y2,score])

        return np.array(boxes_output , dtype=np.float32)

class RCNNDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS2, num_classes=2):
        self.weights = weights
        self.model = None
        self.num_classes = num_classes
        self.name = "faster-rcnn"
        self.display_name = "Faster R-CNN"
        self.loaded = False

    def load(self):
        if self.model is not None:
            return

        try:
            ckpt = torch.load(self.weights, map_location="cpu")
            args = ckpt.get("args", {})

            imgsz = args.get("imgsz", 1024)
            max_det = args.get("max_det", DEFAULT_MAX_DET)
            nms = args.get("nms", 0.35)
            trainable_layers = args.get("trainable_layers", 3)

            # NOTE: these anchor sizes (8,32,64,128,256) must match whatever
            # AnchorGenerator the training script that produced best_frcnn.pt
            # actually used. If that script used a different set (e.g.
            # 8,16,32,64,128), this mismatch will silently degrade every
            # Faster R-CNN prediction regardless of single- or multi-scale
            # inference — the anchor generator has to match the checkpoint,
            # not just look plausible.
            anchors = AnchorGenerator(sizes=((8,), (32,), (64,), (128,), (256,)),aspect_ratios=((0.75, 1.0, 1.33),) * 5,)

            # IMPORTANT: the checkpoint's state_dict keys (BatchNorm inside
            # the FPN, a conv-based box head, a two-conv RPN head) only
            # match the v2 architecture built by fasterrcnn_resnet50_fpn_v2.
            # Manually assembling resnet_fpn_backbone(...) + FasterRCNN(...)
            # builds the older v1 architecture, which has structurally
            # different layers (a 2-layer MLP box head, no FPN BatchNorm)
            # and will fail to load this checkpoint's state_dict at all.
            model = fasterrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=None,
                num_classes=self.num_classes,
                min_size=imgsz,
                max_size=imgsz,
                box_detections_per_img=max_det,
                box_nms_thresh=nms,
                box_score_thresh=0.05,
                rpn_pre_nms_top_n_train=2000,
                rpn_post_nms_top_n_train=1000,
                rpn_pre_nms_top_n_test=1000,
                rpn_post_nms_top_n_test=500,
                rpn_batch_size_per_image=512,
                box_batch_size_per_image=1024,
                trainable_backbone_layers=trainable_layers,
            )

            # rpn_anchor_generator can't be passed as a kwarg above — the
            # fasterrcnn_resnet50_fpn_v2 builder already supplies its own
            # default internally and passing another causes:
            #   TypeError: got multiple values for keyword argument 'rpn_anchor_generator'
            # Swapping it in after construction is safe: the generator
            # holds no learnable weights, and this custom generator
            # produces the same anchors-per-location count (1 size x 3
            # aspect ratios = 3) as the default, so the RPN head's conv
            # layers stay shape-compatible.
            model.rpn.anchor_generator = anchors

            model.load_state_dict(ckpt["model"])
            model.to("cpu").eval()

            # Only commit the model once fully loaded and in eval mode,
            # so a failed/partial load never leaves self.model stuck
            # in default (training) mode for subsequent requests.
            self.model = model
            self.loaded = True
        except Exception as e:
            self.model = None
            self.loaded = False
            raise ModelUnavailable(f"Faster R-CNN weights failed to load: {e}") from e

    def ensure_loaded(self):
        if self.model is None:
            self.load()
        return self.model

    def predict(self,image,conf=0.4,imgsz=None,iou=DEFAULT_IOU,max_det=None):
        # imgsz drives multi-scale inference: predict_multiscale() calls
        # this once per scale, and swapping model.transform.min_size/
        # max_size here (then restoring it) is what makes that safe to do
        # in a loop without leaving the model's default scale mutated
        # between calls or between requests.
        self.ensure_loaded()
        original_min_size = self.model.transform.min_size
        original_max_size = self.model.transform.max_size


        if imgsz:
            self.model.transform.min_size = (imgsz,)
            self.model.transform.max_size = imgsz

        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image).convert("L")
            image = to_tensor(pil)
            image = image.repeat(3, 1, 1)

        with torch.no_grad():
            pred = self.model([image.cpu()])[0]

        self.model.transform.min_size = original_min_size
        self.model.transform.max_size = original_max_size

        boxes = pred["boxes"].numpy()
        scores = pred["scores"].numpy()
        labels = pred["labels"].numpy()

        # Diagnostic: see what the model actually produced before any
        # conf filtering. If raw_max is near 0 or there are zero raw
        # boxes at all, the problem is upstream of thresholds (weights/
        # architecture/label convention) — not the conf value being used.
        raw_max = float(scores.max()) if len(scores) else 0.0
        n_label1 = int((labels == 1).sum())
        print(
            f"RCNN raw -> total_boxes={len(scores)} label1_boxes={n_label1} "
            f"max_score={raw_max:.4f} imgsz={imgsz} conf_filter={conf}"
        )

        det = []

        for box, score, label in zip(boxes, scores, labels):

            if label == 1 and score >= conf:
                x1,y1,x2,y2 = box
                det.append([x1,y1,x2,y2,score])
        if not det:
            return np.empty((0,5),dtype=np.float32)
        det = np.array(det,dtype=np.float32)
        det = det[np.argsort(-det[:, 4])]

        if max_det:
            det = det[:max_det]
        return det

def predict_multiscale(detector, image, imgsz, conf=DEFAULT_CONF, iou=DEFAULT_IOU, max_det=DEFAULT_MAX_DET, merge_iou=0.3, scale_list=None):
    # Generic across detectors: works for YOLODetector and RCNNDetector
    # alike, since both implement predict(image, conf, imgsz, iou, max_det)
    # and RCNNDetector.predict() already restores its own transform size
    # after each call, so looping it here is safe.
    #
    # scale_list lets the caller pass an explicit set of scales (e.g.
    # AVAILABLE_SCALES_RCNN, centered on Faster R-CNN's actual training
    # imgsz) instead of deriving one from the global AVAILABLE_SCALES,
    # which was tuned around YOLO's much wider usable scale range.
    if scale_list:
        imgsz_list = scale_list
    else:
        idx = 0
        for i,s in enumerate(AVAILABLE_SCALES):
            if imgsz >= s:
                idx = i
        imgsz_list = AVAILABLE_SCALES[:idx+1]

    all_boxes = []
    for imgsz in imgsz_list:
        detections = detector.predict(image, conf=conf, imgsz=imgsz, iou=iou, max_det=max_det)
        if detections is not None and len(detections) > 0:
            all_boxes.append(detections)

    if len(all_boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    all_boxes = np.concatenate(all_boxes, axis=0)
    boxes_t = torch.tensor(all_boxes[:, :4], dtype=torch.float32)
    scores_t = torch.tensor(all_boxes[:, 4], dtype=torch.float32)
    keep = nms(boxes_t, scores_t, merge_iou)
    merged = all_boxes[keep.numpy()]
    merged = merged[np.argsort(-merged[:, 4])]
    if max_det:
        merged = merged[:max_det]
    return merged