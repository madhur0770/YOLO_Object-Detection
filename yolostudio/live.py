"""Frame-by-frame analysis for the live camera modes (no Streamlit imports, so it is unit-testable)."""
from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field

from . import counting, inference


def analyse_frame(model, frame_bgr, confidence, allowed, roi: counting.Roi, show_centroids: bool = True):
    """Detect objects in one BGR frame; return (annotated RGB frame, detections, total counts, ROI counts)."""
    height, width = frame_bgr.shape[:2]
    detections = inference.detect(model, frame_bgr, confidence, allowed=allowed)
    total = counting.count_labels(detections)
    in_roi = counting.count_in_roi(detections, roi, width, height)
    return inference.annotate(frame_bgr, detections, roi, show_centroids), detections, total, in_roi


_MODELS: dict = {}
_MODELS_LOCK = threading.Lock()


def get_model(weights: str):
    """Load a model once per process, on first use (safe to call from the video thread)."""
    with _MODELS_LOCK:
        if weights not in _MODELS:
            _MODELS[weights] = inference.load_model(weights)
        return _MODELS[weights]


@dataclass
class LiveSession:
    """Shared between Streamlit's script thread and the video-frame thread of the browser camera."""

    model: object = None
    confidence: float = 0.25
    allowed: set | None = None
    roi: counting.Roi = field(default_factory=counting.Roi)
    show_centroids: bool = True
    total: Counter = field(default_factory=Counter)
    in_roi: Counter = field(default_factory=Counter)
    aggregator: counting.Aggregator = field(default_factory=counting.Aggregator)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def configure(self, model, confidence, allowed, roi, show_centroids) -> None:
        """`model` is a loaded model, or a weights name that is loaded lazily when the first frame arrives."""
        with self.lock:
            self.model, self.confidence, self.allowed = model, confidence, allowed
            self.roi, self.show_centroids = roi, show_centroids

    def reset(self) -> None:
        with self.lock:
            self.total, self.in_roi = Counter(), Counter()
            self.aggregator = counting.Aggregator()

    def process(self, frame_bgr):
        """Analyse a frame, record its counts, and return the annotated frame as BGR."""
        import cv2

        with self.lock:
            model, confidence, allowed = self.model, self.confidence, self.allowed
            roi, show = self.roi, self.show_centroids
        if model is None:
            return frame_bgr
        if isinstance(model, str):
            model = get_model(model)
        rgb, detections, total, in_roi = analyse_frame(model, frame_bgr, confidence, allowed, roi, show)
        height, width = frame_bgr.shape[:2]
        with self.lock:
            self.total, self.in_roi = total, in_roi
            self.aggregator.add_frame(detections, roi, width, height)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def snapshot(self):
        with self.lock:
            return Counter(self.total), Counter(self.in_roi), self.aggregator.frames
