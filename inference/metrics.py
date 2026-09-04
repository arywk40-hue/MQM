"""
Occupancy metrics derived from zone assignments.

Converts zone-assigned detections into the four reported values: headcount,
queue count, seat occupancy, and crowd level.

This module is the sole point at which raw detections become reportable
figures. Downstream components (api/ingestion.py, api/read.py,
dashboard/app.py) persist, serve, and render the output of compute_metrics()
without reinterpreting it.

No third-party dependencies; imports and tests without ultralytics.

Output schema. The pydantic models in api/models.py mirror these keys, and the
dashboard reads them directly, so changes require coordination across both:

    {
      "camera_id": "mess_main",
      "headcount": 42,
      "queue_count": 11,
      "seats_total": 64,
      "seats_occupied": 28,
      "seat_occupancy_pct": 43.8,
      "crowd_level": "amber",
      "zone_counts": {"queue": 11, "seating_front": 12, ...},
      "detections_raw": 47,
      "detections_counted": 42
    }

detections_raw and detections_counted are reported separately to expose how
many detections the confidence floor discarded. A sustained divergence between
them indicates a mistuned threshold, which is not otherwise observable from the
dashboard.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from .zones import (
    DEFAULT_REFERENCE_POINT,
    CameraZones,
    ReferencePoint,
    ZoneAssignment,
    assign_zones,
    count_by_zone,
)

CrowdLevel = Literal["green", "amber", "red"]

# Fallback bands, applied only when a camera declares no crowd_thresholds.
# Banded on headcount rather than seat occupancy: an empty hall with all seats
# free is not congested, whereas a large queue against empty seating is.
DEFAULT_CROWD_THRESHOLDS: dict[str, int] = {"amber": 25, "red": 50}


def crowd_level(headcount: int, thresholds: dict[str, int] | None = None) -> CrowdLevel:
    """
    Band a headcount into green / amber / red.

    Bands are half-open and ascending: green below `amber`, amber from
    `amber` up to `red`, red at `red` and above.

    Thresholds are calibrated against detector output rather than ground
    truth, so they require re-tuning whenever the model or its recall changes.
    They are therefore defined in config/zones.json rather than in code.
    """
    bands = thresholds or DEFAULT_CROWD_THRESHOLDS
    amber = bands.get("amber", DEFAULT_CROWD_THRESHOLDS["amber"])
    red = bands.get("red", DEFAULT_CROWD_THRESHOLDS["red"])

    if amber > red:
        raise ValueError(
            f"crowd thresholds out of order: amber={amber} > red={red}. Check config/zones.json."
        )

    if headcount >= red:
        return "red"
    if headcount >= amber:
        return "amber"
    return "green"


def compute_metrics(
    detections: Iterable[dict],
    config: CameraZones,
    frame_size: tuple[int, int] | None = None,
    mode: ReferencePoint = DEFAULT_REFERENCE_POINT,
) -> dict:
    """
    Compute reported metrics for a single frame.

    Args:
        detections: output of inference/model.py. An empty list denotes an
            empty hall and is not an error.
        config: loaded via zones.load_zones().
        frame_size: (width, height) of the source frame, validated against the
            coordinate space of the polygons. See zones.assign_zones().
        mode: bounding-box reference point used for zone membership.

    Callers that already have zone assignments should use
    metrics_from_assignments() instead of re-running the geometry.
    """
    detections = list(detections)
    assignments = assign_zones(detections, config, frame_size=frame_size, mode=mode)
    return metrics_from_assignments(assignments, config, detections_raw=len(detections))


def metrics_from_assignments(
    assignments: list[ZoneAssignment],
    config: CameraZones,
    detections_raw: int | None = None,
    queue_flags: Iterable[bool] | None = None,
    seated_flags: Iterable[bool] | None = None,
) -> dict:
    """Build metrics from zone assignments, optionally refined by classifiers."""
    zone_counts = count_by_zone(assignments, config)

    def refined_count(zone_type: str, flags: Iterable[bool] | None) -> int:
        if flags is None:
            return sum(zone_counts[zone.name] for zone in config.zones_of_type(zone_type))
        values = list(flags)
        if len(values) != len(assignments):
            raise ValueError(
                f"{zone_type} classifier returned {len(values)} predictions for "
                f"{len(assignments)} zone assignments"
            )
        return sum(
            is_positive and assignment.in_zone_type(zone_type, config)
            for assignment, is_positive in zip(assignments, values, strict=True)
        )

    headcount = len(assignments)
    queue_count = refined_count("queue", queue_flags)

    seats_total = config.total_seats
    # Clamped to capacity. Detection noise can otherwise report more occupants
    # than seats, surfacing as an occupancy figure above 100% on the dashboard.
    seats_occupied = min(refined_count("seating", seated_flags), seats_total) if seats_total else 0

    seat_occupancy_pct = round(100.0 * seats_occupied / seats_total, 1) if seats_total else 0.0

    return {
        "camera_id": config.camera_id,
        "headcount": headcount,
        "queue_count": queue_count,
        "seats_total": seats_total,
        "seats_occupied": seats_occupied,
        "seat_occupancy_pct": seat_occupancy_pct,
        "crowd_level": crowd_level(headcount, config.crowd_thresholds),
        "zone_counts": zone_counts,
        "detections_raw": detections_raw if detections_raw is not None else headcount,
        "detections_counted": headcount,
    }


def empty_metrics(config: CameraZones) -> dict:
    """
    Zeroed metrics for a camera with no usable frame.

    Indistinguishable from a genuinely empty hall except at the call site. The
    ingestion layer must mark the camera offline rather than publishing this as
    a live reading, otherwise the dashboard reports a green zero for a failed
    camera. Provided so that path still returns the complete schema.
    """
    return metrics_from_assignments([], config, detections_raw=0)
