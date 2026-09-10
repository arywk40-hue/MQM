"""Decode one camera JPEG and run the real detection-to-metrics pipeline."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .metrics import metrics_from_assignments
from .zones import DEFAULT_CONFIG_PATH, ZoneAssignment, assign_zones, load_zones

if TYPE_CHECKING:
    from .classifiers import PersonAttributes

Detector = Callable[[np.ndarray], list[dict]]
AttributeClassifier = Callable[[np.ndarray, list[ZoneAssignment]], list["PersonAttributes"]]


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
    attribute_classifier: AttributeClassifier | None = None,
) -> dict:
    """Decode, detect, assign zones, and compute the complete metrics schema."""
    image, frame_size = decode_jpeg(image_bytes)
    config = load_zones(camera_id, DEFAULT_CONFIG_PATH)
    if detector is None:
        from .model import detect_people

        detector = detect_people
    detections = detector(image)
    assignments = assign_zones(detections, config, frame_size=frame_size)

    if attribute_classifier is None:
        from .classifiers import attribute_classifiers_enabled, classify_people

        if attribute_classifiers_enabled():
            attribute_classifier = classify_people

    attributes = attribute_classifier(image, assignments) if attribute_classifier else None
    metrics = metrics_from_assignments(
        assignments,
        config,
        detections_raw=len(detections),
        queue_flags=[attribute.is_queue for attribute in attributes] if attributes else None,
        seated_flags=[attribute.is_seated for attribute in attributes] if attributes else None,
    )
    strategy = os.environ.get("QUEUE_ESTIMATOR", "production").strip().lower()
    if strategy == "production":
        return metrics
    if strategy not in {"geometric", "membership", "occlusion"}:
        raise ValueError("QUEUE_ESTIMATOR must be production, geometric, membership, or occlusion")
    from research.estimator import estimate_queue

    research_detections = [
        detection
        for detection in detections
        if float(detection.get("confidence", 0.0)) >= config.min_confidence
    ]
    research_metrics = estimate_queue(image, research_detections, camera_id, frame_size, strategy)
    research_metrics.pop("research_diagnostics", None)
    metrics.update(research_metrics)
    return metrics
