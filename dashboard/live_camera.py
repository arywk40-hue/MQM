"""Browser camera streaming with per-session, bounded inference state."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PosturePrediction:
    label: str
    confidence: float


def classify_postures(
    image: np.ndarray, detections: list[dict[str, Any]]
) -> list[PosturePrediction]:
    """Classify each detected person crop as sitting or standing."""
    if not detections:
        return []

    from inference.classifiers import classify_people
    from inference.zones import ZoneAssignment, reference_point

    assignments = [
        ZoneAssignment(
            bbox=detection["bbox"],
            confidence=float(detection.get("confidence", 0.0)),
            point=reference_point(detection["bbox"]),
        )
        for detection in detections
    ]
    attributes = classify_people(image, assignments)
    return [
        PosturePrediction(
            label="Sitting" if attribute.is_seated else "Standing",
            confidence=attribute.seated_confidence,
        )
        for attribute in attributes
    ]


class LiveDetector:
    """Analyze at most two frames per second; retain only the latest result."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process_lock = threading.Lock()
        self.last_attempt = float("-inf")
        self.latest: Any = None
        self.state: dict[str, Any] = {}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def reset(self) -> None:
        with self.process_lock:
            with self.lock:
                self.state = {}
                self.latest = None
                self.last_attempt = float("-inf")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        with self.process_lock:
            now = time.monotonic()
            if now - self.last_attempt >= 0.5:
                self.last_attempt = now
                try:
                    from inference.model import detect_people

                    image = frame.to_ndarray(format="bgr24")
                    detections = detect_people(image)
                    posture_error = None
                    try:
                        postures = classify_postures(image, detections)
                    except Exception:
                        logger.exception("Live posture classification failed")
                        postures = []
                        posture_error = (
                            "Posture classification is unavailable; person counting continues."
                        )
                    for index, detection in enumerate(detections, 1):
                        x1, y1, x2, y2 = map(int, detection["bbox"])
                        posture = postures[index - 1] if index <= len(postures) else None
                        colour = (
                            (52, 180, 52)
                            if posture is None or posture.label == "Standing"
                            else (0, 165, 255)
                        )
                        label = str(index)
                        if posture is not None:
                            label = f"{index} {posture.label} {posture.confidence:.0%}"
                        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
                        cv2.putText(
                            image,
                            label,
                            (x1, max(20, y1)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            colour,
                            2,
                        )
                    sitting = sum(posture.label == "Sitting" for posture in postures)
                    standing = sum(posture.label == "Standing" for posture in postures)
                    with self.lock:
                        self.latest = image
                        self.state = {
                            "headcount": len(detections),
                            "sitting": sitting,
                            "standing": standing,
                            "posture_available": len(postures) == len(detections),
                            "posture_error": posture_error,
                            "seconds": time.monotonic() - now,
                            "updated": time.monotonic(),
                        }
                except Exception:
                    logger.exception("Live camera analysis failed")
                    with self.lock:
                        self.latest = None
                        self.state = {
                            "error": "Camera analysis failed. Stop and restart the camera."
                        }
            with self.lock:
                output = self.latest
            result = (
                av.VideoFrame.from_ndarray(output, format="bgr24") if output is not None else frame
            )
            result.pts = frame.pts
            result.time_base = frame.time_base
            return result


def render_live_camera() -> None:
    st.subheader("Phone camera")
    st.caption(
        "Tap START and allow camera access. Keep this page open to count people and classify "
        "sitting or standing continuously."
    )
    st.info("On your phone, open the HTTPS address. On this Mac, localhost also works.")
    st.caption(
        "Video is sent to this Mac for analysis. Audio is off. Frames and results are not saved. "
        "Queue membership still requires a fixed, calibrated camera viewpoint."
    )
    if "live_posture_detector" not in st.session_state:
        st.session_state["live_posture_detector"] = LiveDetector()
    detector = st.session_state["live_posture_detector"]
    ctx = webrtc_streamer(
        key="phone-live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": []},
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": "environment"},
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 5, "max": 10},
            },
            "audio": False,
        },
        video_frame_callback=detector.recv,
        on_video_ended=detector.reset,
        async_processing=True,
    )

    @st.fragment(run_every=1)
    def show_count() -> None:
        if not ctx.state.playing:
            st.caption("Camera stopped. Tap START to begin.")
            return
        state = detector.snapshot()
        if state.get("error"):
            st.error(state["error"])
        elif not state:
            st.info("Waiting for the first frame. Initial model loading can take a few seconds.")
        elif time.monotonic() - state["updated"] > 5:
            st.warning("No recent analysis. Check the camera connection or stop and restart.")
        else:
            with st.container(horizontal=True):
                st.metric("People", state["headcount"], border=True)
                st.metric("Sitting", state.get("sitting", 0), border=True)
                st.metric("Standing", state.get("standing", 0), border=True)
            if state.get("posture_error"):
                st.warning(state["posture_error"])
            st.caption(f"Analysis: {state['seconds']:.2f}s · updates up to twice per second")

    show_count()
    st.caption("Use STOP to release the camera. Keep both devices on the same Wi-Fi.")
