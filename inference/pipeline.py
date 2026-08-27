"""Decode one camera JPEG and run the real detection-to-metrics pipeline."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from .metrics import compute_metrics
from .zones import DEFAULT_CONFIG_PATH, load_zones

Detector = Callable[[np.ndarray], list[dict]]


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a decodable JPEG image."""


def decode_jpeg(image_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """Return a BGR ndarray and its (width, height), rejecting non-JPEG data."""
    if not image_bytes:
        raise InvalidImageError("uploaded image is empty")
    if not image_bytes.startswith(b"\xff\xd8\xff"):
        raise InvalidImageError("expected JPEG bytes")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError("uploaded file is not a valid JPEG")
    height, width = image.shape[:2]
    return image, (width, height)


def process_frame(
    image_bytes: bytes,
    camera_id: str,
    detector: Detector | None = None,
) -> dict:
    """Decode, detect, assign zones, and compute the complete metrics schema."""
    image, frame_size = decode_jpeg(image_bytes)
    config = load_zones(camera_id, DEFAULT_CONFIG_PATH)
    if detector is None:
        from .model import detect_people

        detector = detect_people
    detections = detector(image)
    return compute_metrics(detections, config, frame_size=frame_size)
