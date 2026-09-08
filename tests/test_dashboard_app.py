from __future__ import annotations

from pathlib import Path

import requests
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def run_app(monkeypatch, base_url: str, get):
    monkeypatch.setenv("READ_API_BASE_URL", base_url)
    monkeypatch.setattr(requests, "get", get)
    app = AppTest.from_file(APP).run(timeout=20)
    return app.segmented_control[0].set_value("Live cameras").run(timeout=20)


def test_upload_mode_does_not_require_api(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Photo upload must not request live camera status")

    monkeypatch.setattr(requests, "get", fail)
    app = AppTest.from_file(APP).run(timeout=20)
    assert not app.exception
    assert len(app.get("file_uploader")) == 1
    assert len(app.selectbox) == 1
    assert app.button[0].label == "Analyze image"
    assert app.button[0].disabled
    assert not app.error


def test_dashboard_renders_offline_camera(monkeypatch):
    def get(url, timeout):
        assert url == "http://offline-api/status"
        return Response(
            {"cameras": {"mess_main": {"camera_id": "mess_main", "online": False, "metrics": None}}}
        )

    app = run_app(monkeypatch, "http://offline-api", get)
    assert not app.exception
    assert [element.value for element in app.title] == ["Mess Congestion"]
    assert [element.value for element in app.subheader] == ["Mess Main"]
    assert "Camera offline" in app.error[0].value


def test_dashboard_renders_metrics_and_history_chart(monkeypatch):
    metrics = {
        "camera_id": "mess_main",
        "timestamp": "2026-08-27T12:00:00Z",
        "headcount": 12,
        "queue_count": 3,
        "seats_total": 64,
        "seats_occupied": 8,
        "seat_occupancy_pct": 12.5,
        "crowd_level": "amber",
        "zone_counts": {},
        "detections_raw": 12,
        "detections_counted": 12,
    }

    def get(url, timeout):
        if url == "http://online-api/status":
            return Response(
                {
                    "cameras": {
                        "mess_main": {
                            "camera_id": "mess_main",
                            "online": True,
                            "metrics": metrics,
                        }
                    }
                }
            )
        assert url == "http://online-api/history/mess_main?minutes=60"
        return Response({"points": [metrics]})

    app = run_app(monkeypatch, "http://online-api", get)
    assert not app.exception
    assert [(element.label, element.value) for element in app.metric] == [
        ("Headcount", "12"),
        ("Queue", "3"),
        ("Seats", "8/64"),
        ("Occupied", "12.5%"),
    ]
    assert len(app.get("plotly_chart")) == 1
    assert "AMBER" in app.markdown[0].value


def test_dashboard_surfaces_read_api_failure(monkeypatch):
    def get(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    app = run_app(monkeypatch, "http://broken-api", get)
    assert not app.exception
    assert "Read API request failed" in app.error[0].value
