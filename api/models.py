"""Strict schemas for the inference, storage, API, and dashboard boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CameraMetrics(BaseModel):
    """One complete reading from one camera."""

    model_config = ConfigDict(extra="forbid")

    camera_id: str
    timestamp: datetime
    headcount: int = Field(ge=0)
    queue_count: int = Field(ge=0)
    seats_total: int = Field(ge=0)
    seats_occupied: int = Field(ge=0)
    seat_occupancy_pct: float = Field(ge=0, le=100)
    crowd_level: Literal["green", "amber", "red"]
    zone_counts: dict[str, int]
    detections_raw: int = Field(ge=0)
    detections_counted: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_relationships(self) -> CameraMetrics:
        if self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        if self.queue_count > self.headcount:
            raise ValueError("queue_count cannot exceed headcount")
        if self.seats_occupied > self.seats_total:
            raise ValueError("seats_occupied cannot exceed seats_total")
        if self.detections_counted > self.detections_raw:
            raise ValueError("detections_counted cannot exceed detections_raw")
        if any(value < 0 for value in self.zone_counts.values()):
            raise ValueError("zone_counts cannot contain negative values")
        expected_occupancy = (
            round(100 * self.seats_occupied / self.seats_total, 1) if self.seats_total else 0.0
        )
        if abs(self.seat_occupancy_pct - expected_occupancy) > 0.05:
            raise ValueError("seat_occupancy_pct does not match occupied/total seats")
        return self


class IngestResponse(BaseModel):
    status: Literal["ok"]
    metrics: CameraMetrics


class CameraStatus(BaseModel):
    camera_id: str
    online: bool
    metrics: CameraMetrics | None = None


class StatusResponse(BaseModel):
    cameras: dict[str, CameraStatus]


class HistoryPoint(BaseModel):
    timestamp: datetime
    headcount: int = Field(ge=0)
    queue_count: int = Field(ge=0)
    seats_occupied: int = Field(ge=0)
    seats_total: int = Field(ge=0)
    seat_occupancy_pct: float = Field(ge=0, le=100)
    crowd_level: Literal["green", "amber", "red"]


class HistoryResponse(BaseModel):
    camera_id: str
    minutes: int
    points: list[HistoryPoint]
