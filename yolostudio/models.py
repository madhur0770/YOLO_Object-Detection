"""The models offered in the app."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    weights: str  # a local path, or a public Ultralytics weight name that is downloaded on first use
    description: str

    @property
    def is_local(self) -> bool:
        return Path(self.weights).suffix == ".pt" and (MODELS_DIR / Path(self.weights).name).is_file()


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "mask",
        "Mask detector (custom-trained YOLO11n)",
        str(MODELS_DIR / "mask-detector-yolo11n.pt"),
        "Two classes (mask, no-mask), fine-tuned from YOLO11n on the public-domain Roboflow mask-wearing dataset.",
    ),
    ModelSpec("oiv7-n", "Nano - Open Images v7 (600 classes)", "yolov8n-oiv7.pt", "Fastest general-purpose model."),
    ModelSpec("oiv7-m", "Bingo (medium) - Open Images v7 (600 classes)", "yolov8m-oiv7.pt", "Balance of speed and accuracy."),
    ModelSpec("oiv7-l", "Large - Open Images v7 (600 classes)", "yolov8l-oiv7.pt", "More accurate, slower."),
    ModelSpec("oiv7-x", "XL (extra large) - Open Images v7 (600 classes)", "yolov8x-oiv7.pt", "Most accurate, slowest."),
    ModelSpec("coco-n", "COCO - YOLO11n (80 classes)", "yolo11n.pt", "Small, fast, everyday objects."),
)


def get_model_spec(key: str) -> ModelSpec:
    for spec in MODELS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown model key: {key}")
