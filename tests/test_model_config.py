from __future__ import annotations

import pytest

import inference.model as model


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confidence_threshold": -0.1}, "confidence_threshold"),
        ({"confidence_threshold": 1.1}, "confidence_threshold"),
        ({"iou_threshold": -0.1}, "iou_threshold"),
        ({"iou_threshold": 1.1}, "iou_threshold"),
        ({"image_size": 0}, "image_size"),
    ],
)
def test_detector_rejects_invalid_tuning_before_model_load(kwargs, message):
    with pytest.raises(ValueError, match=message):
        model.PersonDetector(**kwargs)
