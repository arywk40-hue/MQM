"""
Spatial zone assignment for person detections.

Maps the bounding boxes produced by inference/model.py onto named regions of
the frame defined in config/zones.json.

Contains no model, camera, or storage logic. Pure geometry and configuration,
with no third-party dependencies, so it imports and tests without ultralytics.

Input format (produced by inference/model.py):

    [{"bbox": [x1, y1, x2, y2], "confidence": 0.87}, ...]

Bounding boxes are corner coordinates in original-image pixel space. An empty
list denotes no detections and is not an error condition.

Zone polygons must be defined in that same original-image coordinate space.
A mismatch causes every point-in-polygon test to return false and every
downstream metric to read zero without raising. load_zones() therefore records
the frame dimensions the polygons were authored against, and assign_zones()
rejects frames of a different size unless rescaling is requested explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

Point = tuple[float, float]
Polygon = Sequence[Point]

ReferencePoint = Literal["bottom_center", "centroid"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "zones.json"

# Which point on a bounding box represents the subject's floor position.
#
# bottom_center ((x1+x2)/2, y2) is the floor-contact point, and under
# perspective projection is the correct answer to "which region of the floor
# is this person occupying" — the only question this module asks.
#
# centroid ((x1+x2)/2, (y1+y2)/2) sits at approximately chest height, which
# projects forward of the subject's true floor position. For a seated subject
# this can place the reference point on the table ahead of them, or across a
# zone boundary.
#
# Trade-off: seated subjects have occluded legs, so y2 represents the bottom of
# the visible torso rather than the feet, and is correspondingly less stable.
# Configurable for that reason; re-evaluate against production camera footage.
DEFAULT_REFERENCE_POINT: ReferencePoint = "bottom_center"


class ZoneConfigError(ValueError):
    """Raised for malformed or internally inconsistent zone config."""


@dataclass(frozen=True)
class Zone:
    name: str
    type: str                     # "queue" | "seating" | "entrance" | ...
    polygon: tuple[Point, ...]
    seats: int | None = None      # only meaningful for type == "seating"

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)


@dataclass(frozen=True)
class CameraZones:
    camera_id: str
    frame_width: int
    frame_height: int
    zones: tuple[Zone, ...]
    crowd_thresholds: dict[str, int]
    min_confidence: float = 0.0

    def zones_of_type(self, zone_type: str) -> tuple[Zone, ...]:
        return tuple(z for z in self.zones if z.type == zone_type)

    @property
    def total_seats(self) -> int:
        return sum(z.seats or 0 for z in self.zones_of_type("seating"))

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self.frame_width, self.frame_height)


@dataclass
class ZoneAssignment:
    """One detection, plus the zones its reference point fell inside."""

    bbox: list[float]
    confidence: float
    point: Point
    zones: list[str] = field(default_factory=list)

    def in_zone_type(self, zone_type: str, config: CameraZones) -> bool:
        names = {z.name for z in config.zones_of_type(zone_type)}
        return bool(names & set(self.zones))


# --- Geometry ----------------------------------------------------------


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Ray-casting (crossing-number) test.

    Casts a ray in +x from the point and counts edge crossings; odd means
    inside. The `(yi > y) != (yj > y)` guard excludes horizontal edges and
    counts each vertex once, so a point level with a vertex doesn't
    double-count.

    Boundary points are not deterministically classified. This is inherent to
    the algorithm and acceptable here, as a subject straddling a zone edge by
    a single pixel is genuinely ambiguous.
    """
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def reference_point(
    bbox: Sequence[float],
    mode: ReferencePoint = DEFAULT_REFERENCE_POINT,
) -> Point:
    """Reduce a bbox to the single point used for zone membership."""
    x1, y1, x2, y2 = bbox
    if mode == "bottom_center":
        return ((x1 + x2) / 2.0, y2)
    if mode == "centroid":
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    raise ValueError(f"unknown reference point mode: {mode!r}")


# --- Config loading ----------------------------------------------------


def load_zones(
    camera_id: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> CameraZones:
    """Load and validate one camera's zone config."""
    path = Path(config_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ZoneConfigError(f"zone config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ZoneConfigError(f"zone config is not valid JSON: {path}: {exc}") from exc

    cameras = raw.get("cameras", {})
    if camera_id not in cameras:
        known = ", ".join(sorted(cameras)) or "(none)"
        raise ZoneConfigError(f"no zone config for camera {camera_id!r}. known: {known}")

    cam = cameras[camera_id]
    for key in ("frame_width", "frame_height", "zones"):
        if key not in cam:
            raise ZoneConfigError(f"camera {camera_id!r} is missing required key {key!r}")

    zones: list[Zone] = []
    seen: set[str] = set()
    for entry in cam["zones"]:
        name = entry.get("name")
        if not name:
            raise ZoneConfigError(f"camera {camera_id!r} has a zone with no name")
        if name in seen:
            raise ZoneConfigError(f"camera {camera_id!r} has duplicate zone name {name!r}")
        seen.add(name)

        polygon = [tuple(p) for p in entry.get("polygon", [])]
        if len(polygon) < 3:
            raise ZoneConfigError(
                f"zone {name!r} needs at least 3 points, got {len(polygon)}"
            )

        zone_type = entry.get("type", "other")
        seats = entry.get("seats")
        if zone_type == "seating" and seats is None:
            raise ZoneConfigError(
                f"seating zone {name!r} must declare 'seats' — seat occupancy is "
                f"meaningless without a denominator, and the camera cannot count chairs"
            )

        zones.append(
            Zone(name=name, type=zone_type, polygon=tuple(polygon), seats=seats)
        )

    config = CameraZones(
        camera_id=camera_id,
        frame_width=int(cam["frame_width"]),
        frame_height=int(cam["frame_height"]),
        zones=tuple(zones),
        crowd_thresholds={
            k: v for k, v in cam.get("crowd_thresholds", {}).items()
            if not k.startswith("_")
        },
        min_confidence=float(cam.get("min_confidence", 0.0)),
    )

    _validate_polygons_in_frame(config)
    _warn_on_counting_zone_overlap(config)
    return config


def _validate_polygons_in_frame(config: CameraZones) -> None:
    """
    Catch polygons traced against a different resolution than declared.

    A polygon substantially outside the declared frame almost always indicates
    a coordinate-space error, which would otherwise present as uniformly zero
    metrics with no exception raised.
    """
    w, h = config.frame_size
    for zone in config.zones:
        for x, y in zone.polygon:
            if not (-1 <= x <= w + 1 and -1 <= y <= h + 1):
                raise ZoneConfigError(
                    f"zone {zone.name!r} has point ({x}, {y}) outside the declared "
                    f"frame {w}x{h}. The polygon was probably traced against a "
                    f"different resolution than frame_width/frame_height claim."
                )


def find_zone_overlaps(config: CameraZones) -> list[tuple[str, str]]:
    """
    Zone pairs that appear to overlap, by testing each polygon's vertices
    against the other.

    Intentionally non-exhaustive: two rectangles crossing in a plus formation
    have no vertex inside the other and are not detected. Covers the tracing
    errors observed in practice without introducing a shapely dependency.
    """
    overlaps: list[tuple[str, str]] = []
    for i, a in enumerate(config.zones):
        for b in config.zones[i + 1:]:
            if any(b.contains(p) for p in a.polygon) or any(
                a.contains(p) for p in b.polygon
            ):
                overlaps.append((a.name, b.name))
    return overlaps


def _warn_on_counting_zone_overlap(config: CameraZones) -> None:
    """
    Overlapping zones double-count without raising: a subject standing in the
    intersection of an entrance zone and a seating zone is reported as seated.

    Emitted as a warning rather than an error because overlap between purely
    analytic zones is legitimate. Overlap involving a seating or queue zone
    generally is not, as those are intended to partition the floor area.
    """
    import warnings

    counting = {"queue", "seating"}
    by_name = {z.name: z for z in config.zones}
    for a_name, b_name in find_zone_overlaps(config):
        types = {by_name[a_name].type, by_name[b_name].type}
        if types & counting:
            warnings.warn(
                f"camera {config.camera_id!r}: zones {a_name!r} and {b_name!r} "
                f"overlap, and at least one is a counting zone. A person in the "
                f"overlap is counted in both, so queue_count / seats_occupied "
                f"will be inflated. Retrace the polygons so they partition the "
                f"floor.",
                stacklevel=3,
            )


def rescale(config: CameraZones, frame_size: tuple[int, int]) -> CameraZones:
    """
    Return a copy with polygons scaled to a different frame size.

    Only valid when the new frame is the same scene at a different
    resolution. Aspect-ratio changes mean a different crop or lens, and the
    polygons need retracing rather than scaling.
    """
    new_w, new_h = frame_size
    old_w, old_h = config.frame_size

    old_ar = old_w / old_h
    new_ar = new_w / new_h
    if abs(old_ar - new_ar) > 0.01:
        raise ZoneConfigError(
            f"cannot rescale {old_w}x{old_h} -> {new_w}x{new_h}: aspect ratio "
            f"changed ({old_ar:.3f} -> {new_ar:.3f}). Different crop or lens; "
            f"retrace the polygons against the new frame."
        )

    sx, sy = new_w / old_w, new_h / old_h
    scaled = tuple(
        Zone(
            name=z.name,
            type=z.type,
            polygon=tuple((x * sx, y * sy) for x, y in z.polygon),
            seats=z.seats,
        )
        for z in config.zones
    )
    return CameraZones(
        camera_id=config.camera_id,
        frame_width=new_w,
        frame_height=new_h,
        zones=scaled,
        crowd_thresholds=config.crowd_thresholds,
        min_confidence=config.min_confidence,
    )


# --- Assignment --------------------------------------------------------


def assign_zones(
    detections: Iterable[dict],
    config: CameraZones,
    frame_size: tuple[int, int] | None = None,
    mode: ReferencePoint = DEFAULT_REFERENCE_POINT,
    min_confidence: float | None = None,
) -> list[ZoneAssignment]:
    """
    Map each detection to the zones containing its reference point.

    Args:
        detections: model.py output. [] is normal, not an error.
        config: from load_zones().
        frame_size: (width, height) of the frame these detections came from.
            Checked against the size the polygons were traced against. Pass
            None to skip the check only when you already know they match.
        mode: which bbox point represents the person's position.
        min_confidence: override the per-camera floor from config.

    A detection may land in several zones if they overlap; all are recorded.
    """
    if frame_size is not None and tuple(frame_size) != config.frame_size:
        raise ZoneConfigError(
            f"frame is {frame_size[0]}x{frame_size[1]} but zones for "
            f"{config.camera_id!r} were traced against "
            f"{config.frame_width}x{config.frame_height}. Detections and polygons "
            f"are in different coordinate spaces — every zone check would return "
            f"false and every metric would read zero. Retrace the polygons, or "
            f"call rescale() explicitly if it is the same scene at a new resolution."
        )

    floor = config.min_confidence if min_confidence is None else min_confidence

    assignments: list[ZoneAssignment] = []
    for det in detections:
        confidence = float(det.get("confidence", 0.0))
        if confidence < floor:
            continue

        bbox = det["bbox"]
        point = reference_point(bbox, mode=mode)
        assignments.append(
            ZoneAssignment(
                bbox=list(bbox),
                confidence=confidence,
                point=point,
                zones=[z.name for z in config.zones if z.contains(point)],
            )
        )
    return assignments


def count_by_zone(assignments: Iterable[ZoneAssignment], config: CameraZones) -> dict[str, int]:
    """Per-zone occupancy. Every configured zone appears, including empty ones."""
    counts = {z.name: 0 for z in config.zones}
    for a in assignments:
        for name in a.zones:
            counts[name] += 1
    return counts
