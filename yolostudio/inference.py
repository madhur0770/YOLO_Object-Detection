"""Running a YOLO model and drawing the result.

``detect`` converts Ultralytics results into plain :class:`Detection` objects, so everything
downstream (counting, logging, tests) is independent of the model library. Heavy imports
(``ultralytics``, ``cv2``) happen lazily inside the functions that need them.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from .counting import Detection, Roi

BOX_COLOR = (255, 0, 0)  # RGB
ROI_COLOR = (0, 200, 0)
CENTROID_COLOR = (0, 0, 255)


def load_model(path_or_name: str):
    """Load a YOLO model. Known public weight names are downloaded automatically on first use."""
    from ultralytics import YOLO

    return YOLO(path_or_name)


def _to_list(value) -> list:
    return [] if value is None else list(value.tolist())


def results_to_detections(result, names: dict[int, str] | Sequence[str], allowed: Iterable[str] | None = None) -> list[Detection]:
    """Convert one Ultralytics ``Results`` object into detections, optionally keeping only some classes."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy, conf, cls, ids = _to_list(boxes.xyxy), _to_list(boxes.conf), _to_list(boxes.cls), _to_list(getattr(boxes, "id", None))
    keep = set(allowed) if allowed is not None else None
    detections: list[Detection] = []
    for i, (box, score, class_id) in enumerate(zip(xyxy, conf, cls)):
        label = names[int(class_id)]
        if keep is not None and label not in keep:
            continue
        track_id = int(ids[i]) if i < len(ids) else None
        detections.append(Detection(label, float(score), float(box[0]), float(box[1]), float(box[2]), float(box[3]), track_id))
    return detections


def detect(model, frame, confidence: float = 0.25, track: bool = False, allowed: Iterable[str] | None = None) -> list[Detection]:
    """Run the model on one BGR frame (a numpy array) and return detections."""
    if track:
        results = model.track(frame, conf=confidence, persist=True, verbose=False)
    else:
        results = model(frame, conf=confidence, verbose=False)
    return results_to_detections(results[0], model.names, allowed)


def annotate(frame_bgr, detections: list[Detection], roi: Roi, show_centroids: bool = True):
    """Return an RGB copy of ``frame_bgr`` with boxes, labels, the ROI and optional centroids drawn on it."""
    import cv2

    height, width = frame_bgr.shape[:2]
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    left, top, right, bottom = roi.to_pixels(width, height)
    cv2.rectangle(image, (left, top), (right, bottom), ROI_COLOR, 2)
    for d in detections:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(image, p1, p2, BOX_COLOR, 2)
        text = f"{d.label} {d.confidence:.2f}" + (f" #{d.track_id}" if d.track_id is not None else "")
        cv2.putText(image, text, (p1[0], max(12, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 2)
        if show_centroids and roi.contains(d, width, height):
            cx, cy = d.center
            cv2.circle(image, (int(cx), int(cy)), 5, CENTROID_COLOR, -1)
    return image


def decode_image(data: bytes):
    """Decode image bytes (jpg/png) to a BGR numpy array, or raise ValueError."""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The file could not be decoded as an image.")
    return image
