"""Authenticated camera ingestion: JPEG -> inference -> durable + latest storage."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile, status
from starlette.concurrency import run_in_threadpool

from api.auth import require_camera_key
from api.cache import invalidate_read_cache
from api.config import get_settings
from api.models import CameraMetrics, IngestResponse
from api.storage.influx_client import write_metrics
from api.storage.redis_client import set_latest
from inference.pipeline import InvalidImageError, process_frame
from inference.zones import ZoneConfigError

logger = logging.getLogger(__name__)
router = APIRouter()


def _persist_reading(camera_id: str, metrics: CameraMetrics) -> None:
    payload = metrics.model_dump(mode="json")
    write_metrics(camera_id, payload, timestamp=metrics.timestamp)
    set_latest(camera_id, payload)
    invalidate_read_cache()


@router.post(
    "/ingest/{camera_id}",
    response_model=IngestResponse,
    dependencies=[Depends(require_camera_key)],
)
async def ingest_image(
    camera_id: Annotated[str, Path(description="Configured camera ID, e.g. mess_main")],
    file: Annotated[UploadFile, File(description="JPEG snapshot from the camera")],
) -> IngestResponse:
    settings = get_settings()
    if camera_id not in settings.known_cameras:
        raise HTTPException(status_code=404, detail=f"Unknown camera_id: {camera_id}")
    if file.content_type not in {"image/jpeg", "image/jpg"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG uploads are accepted",
        )

    image_bytes = await file.read(settings.max_upload_bytes + 1)
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded image exceeds MAX_UPLOAD_BYTES")

    try:
        raw_metrics = await run_in_threadpool(process_frame, image_bytes, camera_id)
        metrics = CameraMetrics(timestamp=datetime.now(UTC), **raw_metrics)
    except (InvalidImageError, ZoneConfigError, ValueError) as exc:
        logger.info("Rejected frame from %s: %s", camera_id, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed for camera_id=%s", camera_id)
        raise HTTPException(status_code=503, detail="Inference is unavailable") from exc
    finally:
        await file.close()

    try:
        await run_in_threadpool(_persist_reading, camera_id, metrics)
    except Exception as exc:
        logger.exception("Storage failed for camera_id=%s", camera_id)
        raise HTTPException(
            status_code=503,
            detail="Reading could not be persisted; retry this frame",
        ) from exc

    return IngestResponse(status="ok", metrics=metrics)
