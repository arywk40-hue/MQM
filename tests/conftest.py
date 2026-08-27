from __future__ import annotations

import pytest

from api.cache import invalidate_read_cache
from api.config import reset_settings_cache


@pytest.fixture
def configured_env(monkeypatch):
    values = {
        "REDIS_URL": "redis://unused:6379/0",
        "INFLUX_URL": "http://unused:8086",
        "INFLUX_TOKEN": "test-token",
        "INFLUX_ORG": "test-org",
        "INFLUX_BUCKET": "test-bucket",
        "CAMERA_API_KEY": "test-camera-key",
        "KNOWN_CAMERAS": "mess_main",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    reset_settings_cache()
    invalidate_read_cache()
    yield values
    reset_settings_cache()
    invalidate_read_cache()
