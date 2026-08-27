"""Read-only Streamlit dashboard for current congestion and recent trends."""

from __future__ import annotations

import os

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

from dashboard.client import DashboardAPIError, fetch_history, fetch_status

load_dotenv()

CROWD_COLOURS = {"green": "#16a34a", "amber": "#d97706", "red": "#dc2626"}


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
