"""Fine-tune a YOLO model on a dataset in YOLO format.

Example (the configuration used for the bundled mask detector):
    python scripts/train.py --data datasets/mask-wearing/data.yaml --model yolo11n.pt --epochs 100 --imgsz 640
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="path to the dataset's data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="starting weights (downloaded if it is a public name)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU index such as 0, or 'cpu'")
    parser.add_argument("--workers", type=int, default=0, help="0 is the safest setting on Windows")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data.is_file():
        raise SystemExit(f"Dataset file not found: {args.data}. See scripts/download_dataset.py.")
    from ultralytics import YOLO  # imported late so that --help works without the dependency

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers
    )
    print(f"Training complete. Best weights: {results.save_dir}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
