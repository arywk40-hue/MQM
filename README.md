# Mess Queue Management

Mess Queue Management is a camera-based congestion-monitoring system for a college mess. Raspberry Pi cameras upload JPEG frames; the server detects people, estimates queue and seating use, stores live and historical metrics, and serves a read-only Streamlit dashboard.

This README documents the implemented software, not only the original project proposal. Status is current as of 4 September 2026.

## What is working

- Authenticated camera-frame ingestion with FastAPI.
- Real YOLOv8 person detection in original frame coordinates.
- Camera-specific polygon zones for queue, seating, and entrance regions.
- Reviewed MobileNetV2 classifiers that refine queue and seated counts from detected person crops.
- Redis latest-reading cache with camera TTL/offline state.
- InfluxDB time-series history for charts.
- Read-only Streamlit dashboard with current metrics and a 60-minute trend.
- Docker Compose local storage, reproducible commands, API tests, classifier tests, dashboard tests, and an opt-in real-stack test.

## System architecture

```text
Raspberry Pi camera
  │ authenticated JPEG POST
  ▼
FastAPI ingestion API
  ├─ OpenCV JPEG decoding and frame-size validation
  ├─ YOLOv8 person detection
  ├─ queue/seating/entrance zone assignment
  ├─ MobileNetV2 queue + seated crop classifiers
  ├─ InfluxDB: durable timestamped history
  └─ Redis: latest reading with expiry

Streamlit dashboard
  ├─ GET /status       → current online/offline state
  └─ GET /history/{id} → recent time-series trend
```

The dashboard never receives database credentials or connects directly to Redis/InfluxDB. Ingestion succeeds only after both storage writes succeed; inference and storage failures are returned explicitly.

## Software delivered

### Computer vision and metrics

- inference/model.py loads YOLOv8 and returns COCO person detections only.
- inference/pipeline.py decodes JPEG bytes, validates camera coordinate space, and runs the full inference-to-metrics path.
- inference/zones.py maps person floor points to configured polygons and rejects incompatible frame dimensions instead of producing false zeroes.
- inference/metrics.py produces headcount, queue count, occupied seats, occupancy percentage, crowd level, per-zone counts, and detection totals.
- inference/classifiers.py safely loads two reviewed tensor-only MobileNetV2 state dictionaries and classifies YOLO person crops in a batch.

Queue classification applies only to people already in a queue zone. Seated classification applies only to people in a seating zone. This combines visual posture/context with the physical layout rather than allowing a classifier to invent a queue or seat anywhere in the frame.

### Backend and storage

- POST /ingest/{camera_id} accepts authenticated multipart JPEG frames.
- X-Camera-Key authentication, upload-size checks, JPEG validation, unknown-camera rejection, and strict Pydantic metric invariants are enforced.
- Redis stores expiring current snapshots; offline cameras return online: false rather than fake zero values.
- InfluxDB stores complete readings and returns sorted historical metrics.
- GET /status, GET /status/{camera_id}, and GET /history/{camera_id} are read-only endpoints.
- /health is process liveness; /ready checks Redis and InfluxDB and returns HTTP 503 until both are available.

### Dashboard

- dashboard/app.py provides a read-only Streamlit dashboard.
- It refreshes current state every eight seconds and shows headcount, queue, seats, occupancy, crowd level, timestamp, and a trend chart.
- It renders explicit offline, configuration, API-error, and no-history states instead of using mock readings.

### Operations and quality tooling

- docker-compose.yml runs Redis and InfluxDB locally.
- scripts/dev.sh starts storage, FastAPI, and Streamlit together.
- Makefile supplies install, test, lint, type-check, format-check, and storage lifecycle commands.
- .env.example and .streamlit/secrets.toml.example document required variables without committing real credentials.
- tools/load_test.py provides concurrent endpoint load testing.

## Pull-request history

| PR | Status | Delivered work |
| --- | --- | --- |
| [#1](https://github.com/the-robotronics-club/MQM/pull/1) | Closed | Initial infrastructure proposal; superseded by the integrated implementation. |
| [#2](https://github.com/the-robotronics-club/MQM/pull/2) | Merged | Real JPEG-to-YOLO-to-zone metrics pipeline and regression tests. |
| [#3](https://github.com/the-robotronics-club/MQM/pull/3) | Merged | Authenticated API, Redis current state, Influx history, readiness checks, and API tests. |
| [#4](https://github.com/the-robotronics-club/MQM/pull/4) | Merged | Read-only Streamlit dashboard, API client, current metrics, trend chart, and dashboard tests. |
| [#5](https://github.com/the-robotronics-club/MQM/pull/5) | Merged | Developer workflow, quality tooling, load test, Compose validation, and implementation documentation. |
| [#6](https://github.com/the-robotronics-club/MQM/pull/6) | Open | Queue/seated MobileNetV2 classifier integration, reviewed weights, configuration, provenance note, and focused tests. |

## Local setup

Requirements: Python 3.11+, Docker Desktop/Engine with Compose, and curl.

```bash
make install
cp .env.example .env
```

Edit .env and replace every change-me value. At minimum, set:

- CAMERA_API_KEY — secret used by camera uploads.
- INFLUX_TOKEN — local InfluxDB or production Cloud token.
- INFLUX_INIT_PASSWORD — local InfluxDB bootstrap password.

Keep the reviewed classifiers enabled unless diagnosing or retraining:

```dotenv
ATTRIBUTE_CLASSIFIERS_ENABLED=true
QUEUE_CLASSIFIER_WEIGHTS=queue_classifier.pt
SEATED_CLASSIFIER_WEIGHTS=seated_classifier.pt
```

Start the complete application:

```bash
make dev
```

- Dashboard: http://127.0.0.1:8501
- API docs: http://127.0.0.1:8000/docs
- Liveness: http://127.0.0.1:8000/health
- Readiness: http://127.0.0.1:8000/ready

Press Ctrl-C to stop FastAPI and Streamlit. Stop local storage when finished:

```bash
make services-down
```

## Exercise the real flow

With make dev running, upload the committed frame that matches the active 736×490 zone geometry:

```bash
set -a; source .env; set +a

curl --fail --request POST http://127.0.0.1:8000/ingest/mess_main \
  --header "X-Camera-Key: $CAMERA_API_KEY" \
  --form "file=@data/samples/mess_hall_dense.jpg;type=image/jpeg"

curl --fail http://127.0.0.1:8000/status
curl --fail http://127.0.0.1:8000/status/mess_main
curl --fail "http://127.0.0.1:8000/history/mess_main?minutes=60"
```

Expected failure behavior:

- Unknown cameras return HTTP 404.
- Missing/invalid camera keys return HTTP 401.
- Non-JPEG, corrupt, oversized, or wrong-resolution frames are rejected.
- Unavailable model or storage dependencies return HTTP 503.
- A camera with no fresh Redis reading reports online: false.

## API contract

| Endpoint | Purpose |
| --- | --- |
| POST /ingest/{camera_id} | Authenticated multipart JPEG ingestion. |
| GET /status | Current state for every configured camera. |
| GET /status/{camera_id} | Current state for one camera, including offline status. |
| GET /history/{camera_id}?minutes=60 | One to 1,440 minutes of time-series history. |
| GET /health | Process liveness. |
| GET /ready | Redis and InfluxDB dependency readiness. |

Each reading contains:

```text
camera_id, timestamp, headcount, queue_count,
seats_total, seats_occupied, seat_occupancy_pct,
crowd_level, zone_counts, detections_raw, detections_counted
```

## Configuration reference

| Variable | Purpose |
| --- | --- |
| CAMERA_API_KEY | Shared secret required for camera ingestion. |
| KNOWN_CAMERAS | Comma-separated IDs defined in config/zones.json. |
| REDIS_URL | Local Redis or managed Upstash URL. |
| LATEST_TTL_SECONDS | Freshness window before a camera becomes offline. |
| INFLUX_URL, INFLUX_TOKEN | InfluxDB 2.x/Cloud connection settings. |
| INFLUX_ORG, INFLUX_BUCKET | InfluxDB write/query destination. |
| READ_API_BASE_URL | API URL used by Streamlit. |
| CORS_ALLOW_ORIGINS | Allowed browser origins; no wildcard default. |
| MODEL_* | YOLO weights, confidence, IoU, and inference-size settings. |
| ATTRIBUTE_CLASSIFIERS_ENABLED | Enables queue/seated crop classification. |
| QUEUE_CLASSIFIER_WEIGHTS, SEATED_CLASSIFIER_WEIGHTS | Reviewed paths under models/. |

For production, put backend storage credentials in hosting environment configuration. Put only READ_API_BASE_URL in Streamlit secrets; never put Redis or InfluxDB credentials in the dashboard.

## Models and training handover

- models/yolov8n.pt is the standard YOLO cache. It is ignored and can be downloaded automatically by Ultralytics.
- models/queue_classifier.pt and models/seated_classifier.pt are reviewed production checkpoints tracked by PR #6.
- The raw content/ training handover is ignored. It contains source photos and crop datasets not required at runtime and needs separate privacy/provenance review before publication.
- Architecture, label mapping, hashes, and safe-load verification are in [docs/reviews/2026-09-04-attribute-classifiers.md](docs/reviews/2026-09-04-attribute-classifiers.md).

## Verification commands

```bash
make test
make lint
make typecheck
make format-check
python tools/load_test.py --url http://127.0.0.1:8000/status --requests 1000 --concurrency 100
```

To run the real storage/API test after starting the application, run `make test-live`
in a second terminal. It loads credentials from `.env` and verifies that the
newly uploaded sample reaches both current status and timestamp-matched history.
This writes one sample reading to the configured databases.

For an API running at a different URL:

```bash
BASE_URL=http://127.0.0.1:8000 make test-live
```

Latest verification for PR #6:

- Full suite: **63 passed, 1 opt-in test skipped**.
- Focused classifier/pipeline/metrics suite: **37 passed**.
- Ruff, MyPy, formatting, compilation, shell syntax, and Compose validation: **passed**.
- Isolated real stack with Redis, InfluxDB, FastAPI, YOLO, both classifiers, and Streamlit dashboard: **passed**.

## Repository layout

```text
api/                    FastAPI routes, auth, schemas, cache, storage adapters
dashboard/              Read-only Streamlit UI and HTTP client
inference/              YOLO detection, crop classifiers, zones, metrics
models/                 Reviewed classifiers and local YOLO cache
config/zones.json       Camera geometry, capacities, crowd thresholds
data/samples/           Committed test frames
tests/                  Unit, API, dashboard, pipeline, and live-stack tests
tools/                  Load test, zone drawing, evaluation utilities
docs/                   Handover records, model reviews, implementation notes
```

## Remaining deployment work

- Retrace config/zones.json using frames from final mounted cameras. Current zones are calibrated to a 736×490 sample and deliberately reject other sizes.
- Confirm Pi capture resolution, lens crop, mounting angle, and upload cadence.
- Validate queue/seated classifier accuracy on held-out footage from the real mess before using estimates operationally.
- Provision production Redis/InfluxDB credentials only in the backend host.
- Configure the Raspberry Pi uploader to send authenticated JPEG frames at the desired interval.

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for handover guidance and docs/reviews/ for model/integration evidence.
