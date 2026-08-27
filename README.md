# Mess Queue Management System - Core Team Proposal

## 1. Project Overview & Objectives

**Project Duration:** Mid-August through September

**Goal:** Address the lack of real-time visibility into mess congestion and reduce peak-hour overcrowding.

**Solution:** A live dashboard displaying real-time queue size, seat occupancy, headcount, and crowd level to enable students to make informed dining decisions.

**Scope:** Deployment of two (2) sensing units to cover the full mess area.(though we will start with 1 initially)

## 2. System Architecture

**Edge Sensors (Perception Layer):** 2x Raspberry Pi Zero boards paired with Pi Camera modules. These will capture high-quality snapshots periodically (e.g., every 10 seconds).

**Enclosure & Power:** Housed in custom 3D-printed enclosures and powered via standard USB adapters, with power-backup considerations.

- **Data Transmission:** Snapshots are transmitted over the IIT Mandi Campus Wi-Fi to a central server with low latency.

**Central Processing (Server):** Receives images and runs YOLOv8 object detection inference to estimate crowd metrics and queue length. Data is stored in a Time-Series Database (e.g., InfluxDB).

**User Interface:** A read-only Streamlit dashboard fetching data only through the Read API.

## 3. Team Allotment & Responsibilities

### Web Development & Computer Vision (Web-D + CV)

**Team Members:** Curio, Agam, Saiprasanth, Ariyan, Ishan

- **Computer Vision Pipeline:** Implement and optimize the YOLOv8 object detection inference on the central server for accurate headcount, seat occupancy, and queue length estimation.

- **Backend Server:** Set up the API Gateway (FastAPI/Node.js) and Time-Series Database (InfluxDB) to receive and log data from the Pi units.

- **Web Dashboard:** Develop the frontend (React/Next.js) to provide a seamless, live UI for students to check mess congestion.

### RPi Power Management (BMS, Backup)

#### **Team Member:** Ritisha

- **Power Delivery System:** Manage the deployment of standard USB adapters for the sensing units.

- **Backup & BMS:** Design and integrate a Battery Management System (BMS) or UPS backup to ensure continuous operation during short power fluctuations and enable safe shutdowns.

- **Power Optimization:** Monitor and validate the thermal and power draw of the edge units running 24/7.(see what we can do about thermal power dissipation though not that imp for now)

### RPi OS & Programming

#### **Team Member:** Kartik (Lead)

- **System Configuration:** Flash and configure the Raspberry Pi Zero OS for headless, stable operation on the campus Wi-Fi network.

- **Capture & Transmission Scripts:** Write lightweight Python scripts utilizing `libcamera` to wake up, capture JPEGs, and reliably publish them via HTTP POST or MQTT to the central server every 10 seconds.

- **Edge Reliability:** Implement watchdogs and auto-restart mechanisms to ensure the edge scripts recover from network drops.

### Mechanical

**Team Members:** Suman, Priyanka

- **Enclosure Design:** CAD design and 3D print custom enclosures to house the Pi Zero, Camera, and power modules securely.

- **Thermal & Mounting:** Ensure the enclosure allows for passive cooling and provides the correct mounting angles for optimal camera Field of View (FOV).

- **Physical Deployment:** Securely mount the 2 sensing units at the identified vantage points in the mess hall.

---

# Development

## Working architecture

```text
Pi camera -- authenticated JPEG POST --> FastAPI ingestion router
  --> YOLOv8 person detection --> configured zone assignment --> metrics
  --> InfluxDB (timestamped history)
  --> Redis (latest reading with an expiry)

Streamlit dashboard --> FastAPI read router
  --> Redis for current/offline state
  --> InfluxDB for the last-hour trend
```

The dashboard never connects to Redis or InfluxDB. Ingestion returns success
only after both stores accept the reading; storage and inference failures are
returned explicitly instead of being hidden behind generated data.

## Local setup and startup

Requirements: Python 3.11+, Docker Desktop/Engine with Compose, and `curl`.

```bash
make install
cp .env.example .env
```

Edit `.env` and replace both `change-me` values. `CAMERA_API_KEY` authenticates
camera uploads; `INFLUX_TOKEN` and `INFLUX_INIT_PASSWORD` initialize the local
InfluxDB container. Do not commit `.env`.

Start the complete application:

```bash
make dev
```

This waits for Redis and InfluxDB, starts the API at
`http://127.0.0.1:8000` and Streamlit at `http://127.0.0.1:8501`, and stops the
two application processes on Ctrl-C. Run `make services-down` when the local
database containers are no longer needed.

Ultralytics downloads `yolov8n.pt` on first use. To avoid that download, place
the weight at `models/yolov8n.pt`; model weights are intentionally ignored by
Git.

## Exercise the real flow

With `make dev` running, send the committed 736x490 sample that matches the
current zone coordinate space:

```bash
set -a; source .env; set +a
curl --fail --request POST http://127.0.0.1:8000/ingest/mess_main \
  --header "X-Camera-Key: $CAMERA_API_KEY" \
  --form "file=@data/samples/mess_hall_dense.jpg;type=image/jpeg"
curl --fail http://127.0.0.1:8000/status
curl --fail http://127.0.0.1:8000/status/mess_main
curl --fail "http://127.0.0.1:8000/history/mess_main?minutes=60"
```

`GET /health` is process liveness. `GET /ready` checks Redis and InfluxDB and
returns HTTP 503 until both are available. Interactive API documentation is at
`/docs`.

## API contract

- `POST /ingest/{camera_id}` accepts only a multipart JPEG and requires the
  `X-Camera-Key` header. It rejects unknown cameras, corrupt files, coordinate
  mismatches, oversized uploads, and unavailable dependencies.
- `GET /status` reports every configured camera, including an explicit
  `online: false` state when its Redis TTL expires.
- `GET /status/{camera_id}` returns one current status.
- `GET /history/{camera_id}?minutes=60` returns 1-1440 minutes of timestamped
  history from InfluxDB.

The stable metric fields are `camera_id`, `timestamp`, `headcount`,
`queue_count`, `seats_total`, `seats_occupied`, `seat_occupancy_pct`,
`crowd_level`, `zone_counts`, `detections_raw`, and `detections_counted`.

## Configuration

| Variable | Purpose |
|---|---|
| `CAMERA_API_KEY` | Shared secret required only for camera ingestion |
| `KNOWN_CAMERAS` | Comma-separated IDs that must also exist in `config/zones.json` |
| `REDIS_URL` | Redis/Upstash connection URL (`rediss://` is supported) |
| `LATEST_TTL_SECONDS` | How long a camera remains online without a new frame |
| `INFLUX_URL`, `INFLUX_TOKEN` | InfluxDB 2.x/Cloud endpoint and token |
| `INFLUX_ORG`, `INFLUX_BUCKET` | InfluxDB write/query destination |
| `READ_API_BASE_URL` | API URL used by Streamlit; set in Streamlit Cloud secrets in production |
| `CORS_ALLOW_ORIGINS` | Comma-separated browser origins, never wildcarded by default |
| `MODEL_*` | Optional weights, confidence, IoU, and image-size overrides |

For production, replace the local Redis URL with the managed Upstash URL and
the Influx values with the Cloud values. Put backend values in Render/Railway
environment settings and `READ_API_BASE_URL` in Streamlit Cloud secrets. Never
put service credentials in Streamlit: the dashboard only needs the public Read
API URL.

## Quality commands

```bash
make test
make lint
make typecheck
make format-check
python tools/draw_zones.py data/samples/mess_hall_dense.jpg mess_main
python tools/benchmarks/resolution_sweep.py data/samples/mess_hall_dense.jpg
python tools/load_test.py --url http://127.0.0.1:8000/status --requests 1000 --concurrency 100
```

Set `RUN_LIVE_STACK=1` and `CAMERA_API_KEY` to run the opt-in test against a
running real stack: `pytest tests/test_live_stack.py -q`.

## Layout

```text
api/                    FastAPI app, auth, schemas, read/ingestion routers, storage adapters
dashboard/              read-only Streamlit UI and HTTP client
inference/              JPEG pipeline, YOLO detector, zones, and metrics
config/zones.json       per-camera geometry, capacity, thresholds
tests/                  unit, API, pipeline, dashboard, and opt-in live-stack tests
tools/                  zone drawing, load test, and non-production benchmarks
docker-compose.yml      local Redis + InfluxDB only
data/samples/           committed input images
data/outputs/           ignored generated artifacts
models/                 ignored model weights
docs/                   specification, reviews, and integration workflow
```

## Known external calibration blockers

- The current `mess_main` zone geometry was traced against the committed stock
  736x490 image. It must be retraced from a real mounted-camera frame before
  deployment; the API deliberately rejects a different resolution.
- Camera capture resolution is still unspecified. Existing benchmarks show it
  materially changes recall and the best inference size.
- Seat occupancy remains the documented person-in-seating-zone heuristic, not
  a trained chair-occupancy classifier.

See `docs/INTEGRATION.md` for handovers and `docs/reviews/` for model-selection
evidence and unresolved real-camera validation work.
