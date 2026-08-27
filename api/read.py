"""Read-only endpoints serving Redis current state and InfluxDB history."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Path, Query

from api.cache import get_or_set
from api.config import get_settings
from api.models import (
    CameraMetrics,
    CameraStatus,
    HistoryPoint,
    HistoryResponse,
    StatusResponse,
)
from api.storage.influx_client import query_history
from api.storage.redis_client import get_all_latest, get_latest

logger = logging.getLogger(__name__)
router = APIRouter()


def _require_known_camera(camera_id: str) -> None:
    if camera_id not in get_settings().known_cameras:
        raise HTTPException(status_code=404, detail=f"Unknown camera_id: {camera_id}")


def _camera_status(camera_id: str, raw: dict | None) -> CameraStatus:
    if raw is None:
        return CameraStatus(camera_id=camera_id, online=False, metrics=None)
    return CameraStatus(camera_id=camera_id, online=True, metrics=CameraMetrics(**raw))


@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    settings = get_settings()
    try:
        latest = get_or_set(
            ("all-status", settings.known_cameras),
            settings.read_cache_ttl_seconds,
            lambda: get_all_latest(settings.known_cameras),
        )
        response = StatusResponse(
            cameras={
                camera_id: _camera_status(camera_id, latest.get(camera_id))
                for camera_id in settings.known_cameras
            }
        )
    except Exception as exc:
        logger.exception("Redis status read failed")
        raise HTTPException(status_code=503, detail="Current camera status is unavailable") from exc
    return response


@router.get("/status/{camera_id}", response_model=CameraStatus)
def get_camera_status(camera_id: str = Path()) -> CameraStatus:
    _require_known_camera(camera_id)
    settings = get_settings()
    try:
        raw = get_or_set(
            ("camera-status", camera_id),
            settings.read_cache_ttl_seconds,
            lambda: get_latest(camera_id),
        )
        response = _camera_status(camera_id, raw)
    except Exception as exc:
        logger.exception("Redis status read failed for %s", camera_id)
        raise HTTPException(status_code=503, detail="Current camera status is unavailable") from exc
    return response


@router.get("/history/{camera_id}", response_model=HistoryResponse)
def get_camera_history(
    camera_id: str = Path(),
    minutes: int = Query(60, ge=1, le=1440),
) -> HistoryResponse:
    _require_known_camera(camera_id)
    settings = get_settings()
    try:
        raw = get_or_set(
            ("history", camera_id, minutes),
            settings.read_cache_ttl_seconds,
            lambda: query_history(camera_id, minutes),
        )
        points = [HistoryPoint(**point) for point in raw]
    except Exception as exc:
        logger.exception("InfluxDB history read failed for %s", camera_id)
        raise HTTPException(status_code=503, detail="Camera history is unavailable") from exc
    return HistoryResponse(camera_id=camera_id, minutes=minutes, points=points)
