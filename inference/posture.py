"""Shared posture classification for photos, video, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PosturePrediction:
    label: str
    confidence: float


def classify_postures(
    image: np.ndarray, detections: list[dict[str, Any]]
) -> list[PosturePrediction]:
    """Classify each detected person crop as sitting or standing."""
    if not detections:
        return []

    from inference.classifiers import classify_people
    from inference.zones import ZoneAssignment, reference_point

    assignments = [
        ZoneAssignment(
            bbox=detection["bbox"],
            confidence=float(detection.get("confidence", 0.0)),
            point=reference_point(detection["bbox"]),
        )
        for detection in detections
    ]
    attributes = classify_people(image, assignments)
    return [
        PosturePrediction(
            label="Sitting" if attribute.is_seated else "Standing",
            confidence=attribute.seated_confidence,
        )
        for attribute in attributes
    ]
