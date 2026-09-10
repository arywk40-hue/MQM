"""Photo analysis and live camera congestion dashboard."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

from dashboard.client import DashboardAPIError, fetch_history, fetch_status

load_dotenv()

CROWD_COLOURS = {"green": "#16a34a", "amber": "#d97706", "red": "#dc2626"}
REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_SAMPLE_DIR = REPO_ROOT / "data" / "samples" / "local"
_TEST_FIXTURES = {"mess_hall_dense.jpg", "mess_hall_alt.jpg"}


def local_sample_images() -> list[Path]:
    """Return user-provided local images without exposing test fixtures in the UI."""
    if not LOCAL_SAMPLE_DIR.is_dir():
        return []
    return sorted(
        (
            path
            for path in LOCAL_SAMPLE_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and path.name not in _TEST_FIXTURES
        ),
        key=lambda path: path.name.lower(),
    )


def render_upload() -> None:
    from dashboard.image_analysis import analyze_photo, decode_photo

    st.caption("Upload a photo to count people and see where they were detected. No camera needed.")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="photo_upload",
    )
    samples = local_sample_images()
    selected_sample = st.selectbox(
        "Or choose a local test image",
        options=[None, *samples],
        format_func=lambda path: "Select a local image" if path is None else path.name,
        help=(
            "Images in data/samples/local are only for this machine and are not uploaded "
            "or stored. "
            "Uploads take priority when both are selected."
        ),
    )

    image_data: bytes | None = uploaded.getvalue() if uploaded else None
    image_name: str | None = uploaded.name if uploaded else None
    if image_data is None and selected_sample is not None:
        try:
            image_data = selected_sample.read_bytes()
            image_name = selected_sample.name
        except OSError as exc:
            st.error(f"Could not read {selected_sample.name}: {exc}")
            return

    calibrated = st.toggle(
        "Use saved mess layout for queue and seating estimates",
        help="Only for photos from the same viewpoint and resolution as the saved camera layout.",
    )
    camera_id = None
    if calibrated:
        from inference.zones import DEFAULT_CONFIG_PATH

        cameras = json.loads(DEFAULT_CONFIG_PATH.read_text())["cameras"]
        camera_id = st.selectbox("Saved layout", list(cameras))
        camera = cameras[camera_id]
        st.info(
            f"This layout requires {camera['frame_width']} × {camera['frame_height']} pixels "
            "and the same camera viewpoint. "
            "Current zones and crowd thresholds are sample estimates."
        )

    if image_data is not None:
        try:
            preview = decode_photo(image_data)
        except ValueError as exc:
            st.session_state.pop("photo_result", None)
            st.error(str(exc))
            return
        st.image(preview, caption=image_name, width="stretch")

    analyze = st.button("Analyze image", type="primary", disabled=image_data is None)
    selection = (hashlib.sha256(image_data).hexdigest() if image_data else None, camera_id)
    if st.session_state.get("photo_selection") != selection:
        st.session_state.pop("photo_result", None)
        st.session_state["photo_selection"] = selection

    if analyze and image_data is not None:
        st.session_state.pop("photo_result", None)
        try:
            with st.spinner("Detecting people… The first analysis may take a little longer."):
                analyzed_result = analyze_photo(image_data, camera_id)
                analyzed_result["name"] = image_name
                st.session_state["photo_result"] = analyzed_result
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error(
                "Image analysis is unavailable. Check that the local model weights are installed."
            )
            logging.getLogger(__name__).exception("Photo analysis failed")

    result = st.session_state.get("photo_result")
    if result is None:
        return
    st.subheader(f"Analysis: {result['name']}")
    st.metric("People detected", result["headcount"])
    st.write(
        f"Detected {result['headcount']} people in this "
        f"{result['width']} × {result['height']} image. "
        "Numbered boxes show the detections; small or hidden people may be missed."
    )
    metrics = result["metrics"]
    if metrics is not None:
        queue, seats, occupancy = st.columns(3)
        queue.metric("Queue estimate", metrics["queue_count"])
        seats.metric("Seats occupied", f"{metrics['seats_occupied']}/{metrics['seats_total']}")
        occupancy.metric("Occupancy estimate", f"{metrics['seat_occupancy_pct']:.1f}%")
        st.write(f"Crowd level using the saved sample thresholds: **{metrics['crowd_level']}**.")
    else:
        st.info("Queue, seating, and crowd estimates need a calibrated mess layout for this view.")
    st.image(result["image"], caption="Detected people", width="stretch")
    st.download_button(
        "Download annotated image", result["image"], "people-detected.png", "image/png"
    )
    st.caption(
        "Photo results stay in this session and do not expire or update live camera history."
    )


def api_base_url() -> str:
    try:
        secret_value = st.secrets.get("READ_API_BASE_URL")
    except (FileNotFoundError, KeyError):
        secret_value = None
    value = str(secret_value or os.environ.get("READ_API_BASE_URL", "")).strip()
    if not value:
        raise RuntimeError(
            "READ_API_BASE_URL is not configured. Set it in .env or Streamlit secrets."
        )
    return value.rstrip("/")


@st.cache_data(ttl=5, show_spinner=False)
def cached_status(base_url: str) -> dict:
    return fetch_status(base_url)


@st.cache_data(ttl=5, show_spinner=False)
def cached_history(base_url: str, camera_id: str) -> list[dict]:
    return fetch_history(base_url, camera_id, minutes=60)


def render_camera(base_url: str, camera_id: str, status: dict) -> None:
    st.subheader(camera_id.replace("_", " ").title())
    if not status.get("online") or not status.get("metrics"):
        st.error("Camera offline — no fresh reading was received in the configured TTL window.")
        return

    metrics = status["metrics"]
    level = str(metrics["crowd_level"])
    colour = CROWD_COLOURS.get(level, "#64748b")
    st.markdown(
        f'<div style="display:inline-block;padding:.35rem .7rem;border-radius:999px;'
        f'background:{colour};color:white;font-weight:700">{level.upper()}</div>',
        unsafe_allow_html=True,
    )

    headcount, queue, seats, occupancy = st.columns(4)
    headcount.metric("Headcount", metrics["headcount"])
    queue.metric("Queue", metrics["queue_count"])
    seats.metric("Seats", f"{metrics['seats_occupied']}/{metrics['seats_total']}")
    occupancy.metric("Occupied", f"{metrics['seat_occupancy_pct']:.1f}%")
    st.caption(f"Last update: {metrics['timestamp']}")

    if metrics.get("queue_visible") is not None:
        with st.expander("Research diagnostics"):
            with st.container(horizontal=True):
                st.metric("Visible queue", metrics["queue_visible"], border=True)
                st.metric(
                    "Occlusion correction",
                    f"+{metrics.get('queue_hidden_estimate', 0)}",
                    border=True,
                )
                if metrics.get("queue_ci_low") is not None:
                    st.metric(
                        "Estimated range",
                        f"{metrics['queue_ci_low']}–{metrics['queue_ci_high']}",
                        border=True,
                    )
            st.caption(
                f"Method: {metrics.get('queue_method', 'unknown')} · "
                f"Model: {metrics.get('queue_model_version', 'unknown')}"
            )

    try:
        points = cached_history(base_url, camera_id)
    except DashboardAPIError as exc:
        st.warning(f"History unavailable: {exc}")
        return
    if not points:
        st.info("No history is available yet. It will appear after the first stored readings.")
        return

    figure = px.line(
        points,
        x="timestamp",
        y=["headcount", "queue_count"],
        labels={"value": "People", "timestamp": "Time", "variable": "Metric"},
        title="Last 60 minutes",
    )
    figure.update_layout(margin=dict(l=8, r=8, t=50, b=8), legend_title_text="")
    st.plotly_chart(figure, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Mess Queue", page_icon="🍽️", layout="wide")
    st.title("Mess Congestion")
    mode = st.segmented_control(
        "Mode",
        ["Upload image", "Phone camera", "Live cameras"],
        default="Upload image",
        key="dashboard_mode",
    )
    if mode == "Phone camera":
        from dashboard.live_camera import render_live_camera

        render_live_camera()
        return
    if mode != "Live cameras":
        render_upload()
        return
    st.caption("Live camera estimates. The dashboard is read-only and refreshes every 8 seconds.")
    st_autorefresh(interval=8_000, key="mess-status-refresh")

    try:
        base_url = api_base_url()
        cameras = cached_status(base_url)
    except (RuntimeError, DashboardAPIError) as exc:
        st.error(str(exc))
        st.stop()

    if not cameras:
        st.warning("The API has no configured cameras.")
        return
    for index, (camera_id, status) in enumerate(sorted(cameras.items())):
        if index:
            st.divider()
        render_camera(base_url, camera_id, status)


if __name__ == "__main__":
    main()
