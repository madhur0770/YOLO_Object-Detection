"""Download the public mask-wearing dataset from Roboflow Universe in YOLO format.

The API key is read from the ROBOFLOW_API_KEY environment variable. It is never stored in the repository.
Get a free key at https://app.roboflow.com (Settings -> API Keys).

    PowerShell:  $env:ROBOFLOW_API_KEY = "your-key"; python scripts/download_dataset.py
    bash:        ROBOFLOW_API_KEY=your-key python scripts/download_dataset.py
"""
from __future__ import annotations

import os
import sys

WORKSPACE, PROJECT, VERSION = "joseph-nelson", "mask-wearing", 1


def main() -> int:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("Set the ROBOFLOW_API_KEY environment variable first (see this file's docstring).", file=sys.stderr)
        return 2
    from roboflow import Roboflow  # pip install roboflow

    dataset = Roboflow(api_key=key).workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov8", location="datasets/mask-wearing")
    print(f"Dataset downloaded to: {dataset.location}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
