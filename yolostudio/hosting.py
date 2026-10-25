"""Detect whether the app runs on a hosting service (where the server has no webcam and little memory)."""
from __future__ import annotations

import os
from pathlib import Path


def is_hosted(env=None, app_path: str | None = None) -> bool:
    """True on Streamlit Community Cloud or Hugging Face Spaces.

    Set YOLO_STUDIO_HOSTED=1 (or 0) to force the answer.
    """
    env = os.environ if env is None else env
    forced = env.get("YOLO_STUDIO_HOSTED")
    if forced is not None:
        return forced.strip().lower() in {"1", "true", "yes"}
    if env.get("SPACE_ID"):
        return True
    path = Path(app_path or __file__).as_posix()
    return path.startswith("/mount/src/")
