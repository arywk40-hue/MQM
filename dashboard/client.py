"""HTTP client functions for the read-only dashboard."""

from __future__ import annotations

from typing import Any

import requests


class DashboardAPIError(RuntimeError):
    """Raised when the Read API cannot provide valid data."""


def fetch_json(base_url: str, path: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    try:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise DashboardAPIError(f"Read API request failed for {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DashboardAPIError(f"Read API returned a non-object response for {path}")
    return payload


def fetch_status(base_url: str) -> dict[str, Any]:
    payload = fetch_json(base_url, "/status")
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict):
        raise DashboardAPIError("Read API /status response is missing the cameras object")
    return cameras


def fetch_history(base_url: str, camera_id: str, minutes: int = 60) -> list[dict[str, Any]]:
    payload = fetch_json(base_url, f"/history/{camera_id}?minutes={minutes}")
    points = payload.get("points")
    if not isinstance(points, list):
        raise DashboardAPIError("Read API history response is missing the points list")
    return points
