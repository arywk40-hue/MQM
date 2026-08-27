from __future__ import annotations

import pytest
import requests

from dashboard.client import DashboardAPIError, fetch_history, fetch_json, fetch_status


class Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_fetch_status_and_history(monkeypatch):
    payloads = {
        "http://api/status": {"cameras": {"mess_main": {"online": False}}},
        "http://api/history/mess_main?minutes=60": {"points": []},
    }
    monkeypatch.setattr(requests, "get", lambda url, timeout: Response(payloads[url]))
    assert fetch_status("http://api") == {"mess_main": {"online": False}}
    assert fetch_history("http://api", "mess_main") == []


def test_http_failure_is_not_hidden(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: Response({}, status=503))
    with pytest.raises(DashboardAPIError, match="request failed"):
        fetch_json("http://api", "/status")


def test_malformed_response_is_rejected(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: Response([]))
    with pytest.raises(DashboardAPIError, match="non-object"):
        fetch_json("http://api", "/status")
