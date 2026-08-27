"""Opt-in test against a running API, real Redis/InfluxDB, and real detector."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_STACK") != "1",
    reason="set RUN_LIVE_STACK=1 against the local running stack",
)


def test_live_ingest_read_and_history():
    root = Path(__file__).resolve().parent.parent
    base_url = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
    camera_key = os.environ["CAMERA_API_KEY"]

    ready = requests.get(f"{base_url}/ready", timeout=10)
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"

    with (root / "data" / "samples" / "mess_hall_dense.jpg").open("rb") as image:
        response = requests.post(
            f"{base_url}/ingest/mess_main",
            headers={"X-Camera-Key": camera_key},
            files={"file": ("frame.jpg", image, "image/jpeg")},
            timeout=120,
        )
    assert response.status_code == 200, response.text
    ingested = response.json()["metrics"]

    current = requests.get(f"{base_url}/status/mess_main", timeout=10).json()
    assert current["online"] is True
    assert current["metrics"]["timestamp"] == ingested["timestamp"]

    deadline = time.monotonic() + 15
    points = []
    while time.monotonic() < deadline:
        history = requests.get(f"{base_url}/history/mess_main?minutes=60", timeout=10)
        assert history.status_code == 200, history.text
        points = history.json()["points"]
        if points:
            break
        time.sleep(1)
    assert points
    assert points[-1]["headcount"] == ingested["headcount"]
