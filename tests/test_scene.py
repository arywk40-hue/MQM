from copy import deepcopy

import pytest
from PIL import Image

from inference.annotations import CATEGORY_COLORS, annotate_people
from inference.posture import PosturePrediction
from inference.scene import summarize_scene


def person(x, top=0, bottom=100):
    return {"bbox": [x, top, x + 20, bottom], "confidence": 0.9}


def test_distance_queue_rescue_and_exclusive_counts():
    detections = [person(x) for x in (0, 30, 60, 90, 1000, 2000)]
    original = deepcopy(detections)
    postures = [PosturePrediction("Standing", 0.95)] * 3 + [
        PosturePrediction("Sitting", 0.8),
        PosturePrediction("Sitting", 0.99),
        PosturePrediction("Standing", 0.95),
    ]
    result = summarize_scene(detections, postures)
    assert result["queue_count"] == 4
    assert result["sitting"] == 1
    assert result["standing_other_count"] == 1
    assert result["headcount"] == 6
    assert result["standing"] == 5
    assert result["rescued_count"] == 1
    assert result["queue_groups"] == 1
    assert sum(result[k] for k in ("queue_count", "sitting", "standing_other_count")) == 6
    assert detections == original


def test_two_people_and_round_group_are_not_called_a_queue():
    two = [person(0), person(30)]
    result = summarize_scene(two, [PosturePrediction("Standing", 0.9)] * 2)
    assert result["queue_count"] == 0
    square = [person(0), person(30), person(0, 30, 130), person(30, 30, 130)]
    result = summarize_scene(square, [PosturePrediction("Standing", 0.9)] * 4)
    assert result["queue_count"] == 0
    assert result["rejected_groups"][0]["reason"] == "blob"


def test_confident_seated_person_is_not_rescued():
    result = summarize_scene(
        [person(x) for x in (0, 30, 60, 90)],
        [PosturePrediction("Standing", 0.9)] * 3 + [PosturePrediction("Sitting", 0.99)],
    )
    assert result["queue_count"] == 3
    assert result["sitting"] == 1
    assert result["rescued_count"] == 0


def test_empty_scene_and_invalid_posture_alignment():
    assert summarize_scene([], [])["headcount"] == 0
    with pytest.raises(ValueError, match="one posture"):
        summarize_scene([person(0)], [])


def test_annotations_use_category_colors_without_mutating_source():
    source = Image.new("RGB", (160, 100), "white")
    people = [
        {"bbox": [10 + i * 45, 10, 40 + i * 45, 80], "category": name}
        for i, name in enumerate(("Queue", "Sitting", "Other standing"))
    ]
    result = annotate_people(source, people)
    for i, name in enumerate(("Queue", "Sitting", "Other standing")):
        color = CATEGORY_COLORS[name]
        expected = tuple(int(color[j : j + 2], 16) for j in (1, 3, 5))
        assert result.getpixel((10 + i * 45, 60)) == expected
    assert source.getpixel((10, 60)) == (255, 255, 255)
