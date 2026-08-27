"""Authentication for camera-originated write requests."""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from api.config import get_settings

camera_key_header = APIKeyHeader(name="X-Camera-Key", auto_error=False)


def require_camera_key(provided: str | None = Security(camera_key_header)) -> None:
    expected = get_settings().camera_api_key
    if provided is None or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid camera API key",
        )
