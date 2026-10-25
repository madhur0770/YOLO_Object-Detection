"""Pure-Python detection data model, region-of-interest logic, counting and CSV logging.

Nothing in this module imports a model or an image library, so it is fast to test.
"""
from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    """One detected object. Coordinates are pixels: (x1, y1) top-left, (x2, y2) bottom-right."""

    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


@dataclass(frozen=True)
class Roi:
    """Rectangular region of interest, stored as fractions (0..1) of the frame size."""

    x1: float = 0.15
    y1: float = 0.15
    x2: float = 0.85
    y2: float = 0.85

    def __post_init__(self) -> None:
        for value in (self.x1, self.y1, self.x2, self.y2):
            if not 0.0 <= value <= 1.0:
                raise ValueError("ROI fractions must be between 0 and 1")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("ROI must have x1 < x2 and y1 < y2")

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (round(self.x1 * width), round(self.y1 * height), round(self.x2 * width), round(self.y2 * height))

    def contains(self, detection: Detection, width: int, height: int) -> bool:
        """True when the centre of the detection lies strictly inside the ROI."""
        left, top, right, bottom = self.to_pixels(width, height)
        cx, cy = detection.center
        return left < cx < right and top < cy < bottom


def count_labels(detections: list[Detection]) -> Counter:
    return Counter(d.label for d in detections)


def count_in_roi(detections: list[Detection], roi: Roi, width: int, height: int) -> Counter:
    return Counter(d.label for d in detections if roi.contains(d, width, height))


@dataclass
class Aggregator:
    """Accumulates statistics over many frames (video).

    * ``peak`` - the largest number of objects of each class seen in any single frame
    * ``peak_roi`` - the same, counting only objects whose centre is inside the ROI
    * ``detections`` - how many detections were made in total (an object visible for ten frames counts ten times)
    * ``unique_tracks`` - distinct tracker IDs per class (only filled when tracking is enabled)
    """

    frames: int = 0
    peak: Counter = field(default_factory=Counter)
    peak_roi: Counter = field(default_factory=Counter)
    detections: Counter = field(default_factory=Counter)
    unique_tracks: dict[str, set[int]] = field(default_factory=dict)

    def add_frame(self, detections: list[Detection], roi: Roi, width: int, height: int) -> None:
        self.frames += 1
        frame_counts = count_labels(detections)
        roi_counts = count_in_roi(detections, roi, width, height)
        for label, n in frame_counts.items():
            self.peak[label] = max(self.peak[label], n)
        for label, n in roi_counts.items():
            self.peak_roi[label] = max(self.peak_roi[label], n)
        self.detections.update(frame_counts)
        for d in detections:
            if d.track_id is not None:
                self.unique_tracks.setdefault(d.label, set()).add(d.track_id)

    def labels(self) -> list[str]:
        return sorted(set(self.peak) | set(self.peak_roi) | set(self.detections) | set(self.unique_tracks))


def build_csv(total: Counter, in_roi: Counter) -> str:
    """CSV for a single image or frame: one row per class plus a totals row."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["class", "count", "count_in_roi"])
    for label in sorted(set(total) | set(in_roi)):
        writer.writerow([label, total.get(label, 0), in_roi.get(label, 0)])
    writer.writerow(["TOTAL", sum(total.values()), sum(in_roi.values())])
    return out.getvalue()


def build_video_csv(agg: Aggregator) -> str:
    """CSV for a whole video; see :class:`Aggregator` for what each column means."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["class", "peak_per_frame", "peak_per_frame_in_roi", "total_detections", "unique_tracks"])
    for label in agg.labels():
        tracks = len(agg.unique_tracks[label]) if label in agg.unique_tracks else ""
        writer.writerow([label, agg.peak.get(label, 0), agg.peak_roi.get(label, 0), agg.detections.get(label, 0), tracks])
    writer.writerow(["FRAMES", agg.frames, "", "", ""])
    return out.getvalue()
