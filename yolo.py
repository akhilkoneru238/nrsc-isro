import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

BOX_COLOR = (0, 255, 0)   
BOX_THICKNESS = 2
CONF = 0.4               

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to .pt weights file")
    ap.add_argument("--source", required=True, help="image file or folder of images")
    ap.add_argument("--out", default="detections", help="output folder")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    args, _ = ap.parse_known_args()

    model = YOLO(args.weights)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False,
        max_det = 3000
    )

    for result in results:
        img_path = Path(result.path)
        image = result.orig_img.copy()
        count = 0

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf.item())
                cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                count += 1

        out_path = out_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_path), image)
        print(f"{img_path.name}: {count} crater(s) -> {out_path}")

    print(f"\nDone. Annotated images saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()