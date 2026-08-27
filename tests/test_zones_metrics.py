"""
tests/test_zones_metrics.py

Covers the zone + metrics layer against synthetic detections. No ultralytics,
no camera, no network.

Run:  python -m pytest tests/ -q
      python tests/test_zones_metrics.py     (bare runner, no pytest needed)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.metrics import (  # noqa: E402
    compute_metrics,
    crowd_level,
    empty_metrics,
)
from inference.zones import (  # noqa: E402
    CameraZones,
    Zone,
    ZoneConfigError,
    assign_zones,
    count_by_zone,
    load_zones,
    point_in_polygon,
    reference_point,
    rescale,
)
from tests import fixtures as fx  # noqa: E402

CONFIG = Path(__file__).resolve().parent.parent / "config" / "zones.json"
CAMERA = "mess_main"
FRAME = (736, 490)


def cfg() -> CameraZones:
    return load_zones(CAMERA, CONFIG)


# --- Geometry ----------------------------------------------------------

SQUARE = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))


def test_point_in_polygon_basic():
    assert point_in_polygon((5, 5), SQUARE)
    assert not point_in_polygon((15, 5), SQUARE)
    assert not point_in_polygon((5, 15), SQUARE)
    assert not point_in_polygon((-1, 5), SQUARE)


def test_point_in_polygon_concave():
    # C shape: the notch on the right must read as outside
    c_shape = ((0, 0), (10, 0), (10, 4), (4, 4), (4, 6), (10, 6), (10, 10), (0, 10))
    assert point_in_polygon((2, 5), c_shape)
    assert not point_in_polygon((7, 5), c_shape)
    assert point_in_polygon((7, 2), c_shape)


def test_point_in_polygon_vertex_level_not_double_counted():
    # A point level with a vertex must not flip the crossing count twice
    triangle = ((0, 0), (10, 5), (0, 10))
    assert point_in_polygon((2, 5), triangle)
    assert not point_in_polygon((11, 5), triangle)


def test_point_in_polygon_degenerate():
    assert not point_in_polygon((1, 1), ((0, 0), (1, 1)))
    assert not point_in_polygon((1, 1), ())


def test_reference_point_modes():
    bbox = [100, 200, 200, 400]
    assert reference_point(bbox, "bottom_center") == (150, 400)
    assert reference_point(bbox, "centroid") == (150, 300)


def test_reference_point_rejects_unknown_mode():
    try:
        reference_point([0, 0, 1, 1], "middle_left")  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")


def test_reference_point_choice_changes_zone():
    """
    The reference-point mode is behaviourally significant. A tall subject
    whose feet fall inside a zone but whose torso centre sits above its upper
    edge is assigned differently under each mode.
    """
    zone = Zone(
        name="strip", type="seating", polygon=((0, 100), (100, 100), (100, 200), (0, 200)), seats=4
    )
    bbox = [40, 20, 60, 150]  # feet at y=150 (inside), centroid at y=85 (outside)
    assert zone.contains(reference_point(bbox, "bottom_center"))
    assert not zone.contains(reference_point(bbox, "centroid"))


# --- Config loading ----------------------------------------------------


def test_load_zones():
    c = cfg()
    assert c.camera_id == CAMERA
    assert c.frame_size == FRAME
    assert {z.name for z in c.zones} == {"queue", "seating_front", "seating_mid", "entrance"}
    assert c.total_seats == 64
    assert c.min_confidence == 0.25


def test_load_zones_unknown_camera():
    try:
        load_zones("does_not_exist", CONFIG)
    except ZoneConfigError as exc:
        assert "does_not_exist" in str(exc)
        return
    raise AssertionError("expected ZoneConfigError")


def test_seating_zone_requires_seat_count(tmp_path=None):
    import json
    import tempfile

    bad = {
        "cameras": {
            "x": {
                "frame_width": 100,
                "frame_height": 100,
                "zones": [{"name": "s", "type": "seating", "polygon": [[0, 0], [10, 0], [10, 10]]}],
            }
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(bad, f)
        path = f.name
    try:
        load_zones("x", path)
    except ZoneConfigError as exc:
        assert "seats" in str(exc)
        return
    finally:
        Path(path).unlink(missing_ok=True)
    raise AssertionError("expected ZoneConfigError for seating zone without seats")


def test_polygon_outside_declared_frame_is_rejected():
    import json
    import tempfile

    bad = {
        "cameras": {
            "x": {
                "frame_width": 100,
                "frame_height": 100,
                "zones": [
                    {"name": "q", "type": "queue", "polygon": [[0, 0], [1920, 0], [1920, 1080]]}
                ],
            }
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(bad, f)
        path = f.name
    try:
        load_zones("x", path)
    except ZoneConfigError as exc:
        assert "outside the declared frame" in str(exc)
        return
    finally:
        Path(path).unlink(missing_ok=True)
    raise AssertionError("expected ZoneConfigError for out-of-frame polygon")


# --- The silent-zero failure mode --------------------------------------


def test_frame_size_mismatch_raises_rather_than_reading_zero():
    """
    Polygons in one coordinate space and detections in another causes every
    zone check to return false and every metric to read zero with no exception
    raised. This must fail loudly instead.
    """
    try:
        assign_zones(fx.BUSY, cfg(), frame_size=(1920, 1080))
    except ZoneConfigError as exc:
        assert "coordinate space" in str(exc)
        return
    raise AssertionError("expected ZoneConfigError on frame size mismatch")


def test_rescale_same_aspect_ratio_preserves_membership():
    c = cfg()
    doubled = rescale(c, (1472, 980))
    assert doubled.frame_size == (1472, 980)

    scaled_dets = [
        {"bbox": [v * 2 for v in d["bbox"]], "confidence": d["confidence"]} for d in fx.QUEUE_ONLY
    ]
    before = compute_metrics(fx.QUEUE_ONLY, c, frame_size=FRAME)
    after = compute_metrics(scaled_dets, doubled, frame_size=(1472, 980))
    assert before["queue_count"] == after["queue_count"] == 3


def test_rescale_rejects_aspect_ratio_change():
    try:
        rescale(cfg(), (1000, 1000))
    except ZoneConfigError as exc:
        assert "aspect ratio" in str(exc)
        return
    raise AssertionError("expected ZoneConfigError on aspect ratio change")


# --- Assignment --------------------------------------------------------


def test_empty_detections_yield_zeroes_not_errors():
    m = compute_metrics(fx.EMPTY, cfg(), frame_size=FRAME)
    assert m["headcount"] == 0
    assert m["queue_count"] == 0
    assert m["crowd_level"] == "green"
    assert m["seat_occupancy_pct"] == 0.0


def test_confidence_floor_drops_weak_detections():
    c = cfg()
    m = compute_metrics(fx.LOW_CONFIDENCE, c, frame_size=FRAME)
    assert m["headcount"] == 0
    assert m["detections_raw"] == 2
    assert m["detections_counted"] == 0


def test_people_outside_every_zone_still_count_toward_headcount():
    """Headcount covers all subjects in frame, not only those inside a zone."""
    m = compute_metrics(fx.OUTSIDE_ALL_ZONES, cfg(), frame_size=FRAME)
    assert m["headcount"] == 2
    assert m["queue_count"] == 0
    assert m["seats_occupied"] == 0


def test_count_by_zone_lists_every_zone_including_empty():
    c = cfg()
    counts = count_by_zone(assign_zones(fx.QUEUE_ONLY, c, frame_size=FRAME), c)
    assert set(counts) == {z.name for z in c.zones}
    assert counts["queue"] == 3
    assert counts["seating_front"] == 0


def test_shipped_config_has_no_counting_zone_overlap():
    """
    Overlapping zones double-count without raising: a subject in an
    entrance/seating intersection is reported as seated.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_zones(CAMERA, CONFIG)
    assert not [w for w in caught if "overlap" in str(w.message)], [str(w.message) for w in caught]


def test_overlap_detection_fires_on_overlapping_counting_zones():
    import warnings

    from inference.zones import _warn_on_counting_zone_overlap

    c = CameraZones(
        camera_id="t",
        frame_width=100,
        frame_height=100,
        zones=(
            Zone("seats", "seating", ((0, 0), (60, 0), (60, 60), (0, 60)), seats=4),
            Zone("door", "entrance", ((40, 40), (100, 40), (100, 100), (40, 100))),
        ),
        crowd_thresholds={"amber": 10, "red": 20},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_on_counting_zone_overlap(c)
    assert any("overlap" in str(w.message) for w in caught)


def test_overlapping_zones_record_all_matches():
    c = CameraZones(
        camera_id="t",
        frame_width=100,
        frame_height=100,
        zones=(
            Zone("a", "queue", ((0, 0), (60, 0), (60, 60), (0, 60))),
            Zone("b", "entrance", ((40, 40), (100, 40), (100, 100), (40, 100))),
        ),
        crowd_thresholds={"amber": 10, "red": 20},
    )
    a = assign_zones([fx.person_at(50, 50, height=10)], c, frame_size=(100, 100))
    assert sorted(a[0].zones) == ["a", "b"]


# --- Metrics -----------------------------------------------------------


def test_queue_and_seating_split():
    m = compute_metrics(fx.BUSY, cfg(), frame_size=FRAME)
    assert m["headcount"] == 12
    assert m["queue_count"] == 3
    assert m["seats_occupied"] == 8  # 4 front + 4 mid
    assert m["seats_total"] == 64


def test_seat_occupancy_percentage():
    m = compute_metrics(fx.SEATED_ONLY, cfg(), frame_size=FRAME)
    assert m["seats_occupied"] == 4
    assert m["seat_occupancy_pct"] == round(100 * 4 / 64, 1)


def test_seat_occupancy_clamped_at_capacity():
    """Detection noise must not surface as occupancy above 100%."""
    c = CameraZones(
        camera_id="t",
        frame_width=100,
        frame_height=100,
        zones=(Zone("s", "seating", ((0, 0), (100, 0), (100, 100), (0, 100)), seats=2),),
        crowd_thresholds={"amber": 10, "red": 20},
    )
    dets = [fx.person_at(20 + i * 10, 50, height=10) for i in range(5)]
    m = compute_metrics(dets, c, frame_size=(100, 100))
    assert m["headcount"] == 5
    assert m["seats_occupied"] == 2
    assert m["seat_occupancy_pct"] == 100.0


def test_crowd_level_bands():
    t = {"amber": 25, "red": 50}
    assert crowd_level(0, t) == "green"
    assert crowd_level(24, t) == "green"
    assert crowd_level(25, t) == "amber"  # boundary is inclusive
    assert crowd_level(49, t) == "amber"
    assert crowd_level(50, t) == "red"  # boundary is inclusive
    assert crowd_level(500, t) == "red"


def test_crowd_level_rejects_inverted_thresholds():
    try:
        crowd_level(10, {"amber": 60, "red": 20})
    except ValueError as exc:
        assert "out of order" in str(exc)
        return
    raise AssertionError("expected ValueError for inverted thresholds")


def test_metrics_schema_is_stable():
    """Ingestion and dashboard consume these keys directly."""
    expected = {
        "camera_id",
        "headcount",
        "queue_count",
        "seats_total",
        "seats_occupied",
        "seat_occupancy_pct",
        "crowd_level",
        "zone_counts",
        "detections_raw",
        "detections_counted",
    }
    assert set(compute_metrics(fx.BUSY, cfg(), frame_size=FRAME)) == expected
    assert set(empty_metrics(cfg())) == expected


def test_metrics_are_json_serialisable():
    """Output is persisted to InfluxDB and Redis and returned via the read API."""
    import json

    json.dumps(compute_metrics(fx.BUSY, cfg(), frame_size=FRAME))


# --- Bare runner (no pytest required) ----------------------------------

if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
