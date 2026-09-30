import os

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 40))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import re
from pathlib import Path

import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.transforms import functional as F

CONF_SMALL = 0.10
CONF_LARGE = 0.10

LARGE_PX = 250

GREEN = (0, 255, 0)
ORANGE = (0, 255, 0)

TILE_RE = re.compile(r"x(\d+)_y(\d+)")

def parse_tile_offset(name):

    m = TILE_RE.search(name)

    if m:
        return int(m.group(1)), int(m.group(2))

    return None

def build_model(weights_path, device, imgsz, nms_thresh, max_det):

    anchors = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.75, 1.0, 1.33),) * 5,
    )

    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        num_classes=2,
        min_size=imgsz,
        max_size=imgsz,
        box_detections_per_img=max_det,
        box_nms_thresh=nms_thresh,
        box_score_thresh=0.05,
        rpn_pre_nms_top_n_test=3000,
        rpn_post_nms_top_n_test=1500
    )

    model.rpn.anchor_generator = anchors

    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "model" in checkpoint:

            saved_model = checkpoint["model"]

            if hasattr(saved_model, "state_dict"):
                state_dict = saved_model.state_dict()

            elif isinstance(saved_model, dict):
                state_dict = saved_model

            else:
                raise ValueError(
                    "Unsupported model format inside checkpoint."
                )

        else:

            state_dict = checkpoint

    else:

        if hasattr(checkpoint, "state_dict"):
            state_dict = checkpoint.state_dict()

        else:
            raise ValueError(
                "Unsupported checkpoint format."
            )

    keys = list(state_dict.keys())

    if not any(
        k.startswith(("backbone", "rpn", "roi_heads"))
        for k in keys
    ):
        raise ValueError(
            "\nThe supplied weights do not appear to be Faster R-CNN weights.\n"
            f"First checkpoint keys:\n{keys[:10]}\n\n"
            "Use the .pt/.pth file produced by your Faster R-CNN training."
        )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model

def load_tile_as_tensor(path, device):

    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        return None, None

    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2RGB
    )

    tensor = F.to_tensor(rgb)

    tensor = tensor.to(device)

    return tensor, rgb

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        required=True
    )

    parser.add_argument(
        "--source",
        required=True
    )

    parser.add_argument(
        "--out",
        default="detections"
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Must match --imgsz used during training (single fixed resize)."
    )

    parser.add_argument(
        "--nms",
        type=float,
        default=0.5,
        help="Must match --nms used during training (box_nms_thresh)."
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=1500,
        help="Must match --max-det used during training (box_detections_per_img)."
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Loading weights: {args.weights}")
    print(f"Fixed inference size: {args.imgsz}")
    print(f"Model NMS IoU: {args.nms}")

    model = build_model(
        args.weights,
        device,
        args.imgsz,
        args.nms,
        args.max_det
    )

    print("Faster R-CNN model loaded successfully.")

    src_dir = Path(args.source)
    out_dir = Path(args.out)

    if not src_dir.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {src_dir.resolve()}"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = out_dir / "detections.csv"

    tile_paths = sorted(
        p
        for p in src_dir.iterdir()
        if p.suffix.lower()
        in (
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff"
        )
    )

    print(f"Found {len(tile_paths)} image(s).")

    n_images = 0
    total_craters = 0

    with open(csv_path, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "coords",
            "image",
            "x1",
            "y1",
            "x2",
            "y2",
            "conf",
            "size_px"
        ])

        for img_path in tile_paths:

            tensor, rgb = load_tile_as_tensor(
                img_path,
                device
            )

            if tensor is None:

                print(
                    f"Skip unreadable image: "
                    f"{img_path.name}"
                )

                continue

            n_images += 1

            with torch.inference_mode():

                prediction = model([tensor])[0]

            boxes = prediction["boxes"].detach().cpu().numpy()
            scores = prediction["scores"].detach().cpu().numpy()
            labels = prediction["labels"].detach().cpu().numpy()

            image = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR
            )

            offset = parse_tile_offset(
                img_path.name
            )

            crater_count = 0

            for box, score, label in zip(
                boxes,
                scores,
                labels
            ):

                if int(label) != 1:
                    continue

                x1, y1, x2, y2 = box

                width = x2 - x1
                height = y2 - y1

                size = max(
                    width,
                    height
                )

                conf = float(score)

                if size >= LARGE_PX:
                    threshold = CONF_LARGE
                else:
                    threshold = CONF_SMALL

                if conf < threshold:
                    continue

                crater_count += 1

                x1i = int(x1)
                y1i = int(y1)
                x2i = int(x2)
                y2i = int(y2)

                if size >= LARGE_PX:
                    color = ORANGE
                else:
                    color = GREEN

                cv2.rectangle(
                    image,
                    (x1i, y1i),
                    (x2i, y2i),
                    color,
                    2
                )

                cv2.putText(
                    image,
                    f"{conf:.2f}",
                    (
                        x1i,
                        max(y1i - 6, 10)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )

                if offset:

                    global_x1 = x1i + offset[0]
                    global_y1 = y1i + offset[1]

                    global_x2 = x2i + offset[0]
                    global_y2 = y2i + offset[1]

                    writer.writerow([
                        "strip",
                        img_path.name,
                        global_x1,
                        global_y1,
                        global_x2,
                        global_y2,
                        f"{conf:.3f}",
                        int(size)
                    ])

                else:

                    writer.writerow([
                        "tile",
                        img_path.name,
                        x1i,
                        y1i,
                        x2i,
                        y2i,
                        f"{conf:.3f}",
                        int(size)
                    ])

            total_craters += crater_count

            output_path = (
                out_dir /
                f"pred_{img_path.name}"
            )

            cv2.imwrite(
                str(output_path),
                image
            )

            print(
                f"{img_path.name}: "
                f"{crater_count} crater(s)"
            )

    print()
    print(
        f"Processed {n_images} image(s)."
    )

    print(
        f"Total craters drawn: {total_craters}"
    )

    print(
        f"Results saved to: "
        f"{out_dir.resolve()}"
    )

    print(
        f"CSV saved to: "
        f"{csv_path.resolve()}"
    )

if __name__ == "__main__":
    main()