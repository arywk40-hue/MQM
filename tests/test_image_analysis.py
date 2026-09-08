from __future__ import annotations

import sys
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from dashboard.image_analysis import MAX_IMAGE_BYTES, analyze_photo, decode_photo


def photo_bytes(size=(80, 60), format="PNG"):
    output = BytesIO()
    Image.new("RGB", size, "red").save(output, format=format)
    return output.getvalue()


@pytest.mark.parametrize("data", [b"", b"not an image", b"x" * (MAX_IMAGE_BYTES + 1)])
def test_invalid_upload_is_rejected(data):
    with pytest.raises(ValueError):
        decode_photo(data)


@pytest.mark.parametrize("format", ["JPEG", "PNG"])
def test_photo_analysis_uses_original_resolution_and_bgr(monkeypatch, format):
    def detect(frame):
        assert frame.shape == (60, 80, 3)
        assert frame[0, 0, 2] > 250
        assert frame[0, 0, 0] < 5
        return [{"bbox": [5, 5, 30, 50], "confidence": 0.9}]

    monkeypatch.setitem(sys.modules, "inference.model", SimpleNamespace(detect_people=detect))
    result = analyze_photo(photo_bytes(format=format))
    assert result["headcount"] == 1
    assert result["metrics"] is None
    assert Image.open(BytesIO(result["image"])).size == (80, 60)


def test_wrong_layout_dimensions_rejected_before_detection(monkeypatch):
    def detect(frame):
        raise AssertionError("Detector must not run for incompatible geometry")

    monkeypatch.setitem(sys.modules, "inference.model", SimpleNamespace(detect_people=detect))
    with pytest.raises(ValueError):
        analyze_photo(photo_bytes(), "mess_main")


def test_no_people_is_a_successful_analysis(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "inference.model", SimpleNamespace(detect_people=lambda frame: [])
    )
    result = analyze_photo(photo_bytes())
    assert result["headcount"] == 0
    assert result["detections"] == []
