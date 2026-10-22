# RPi snapshot deployment readiness

The goal is to run inference on a hosted server, receive periodic JPEGs from an
RPi, persist the resulting counts, and update the live dashboard. This supersedes
the earlier "deployment out of scope" instruction.

`FRAME_METRICS_MODE=distance` now connects API ingestion to the tested shared
photo/live scene pipeline. It accepts real snapshot dimensions instead of
requiring the placeholder stock-photo resolution. Calibrated mode retains its
existing zone behavior. Capacity and occupancy stay unavailable in distance
mode, while exclusive queue/sitting/other-standing counts sum to headcount.
New category fields survive Redis status and InfluxDB history roundtrips.

The hosted dashboard opens Live cameras and polls every eight seconds. It shows
the returned counts and reading timestamp. The camera uploader supports current
and older Raspberry Pi capture command names, timeouts, configurable resolution
and interval, failure exit codes, and single-image replay without hardware.

## Real local verification

An isolated Docker Redis/InfluxDB pair and native API/dashboard processes were
used. A real 419 JPEG was sent through the uploader, not an injected detector.
The API returned 35 people, 20 queued, 15 sitting, and zero other standing.
The reading was checked in both current status and persisted history using its
new timestamp. Streamlit AppTest then fetched that same live API and displayed
the same four counts. The running Streamlit health endpoint also returned 200.
After uploads stopped, Redis expiry was checked: the camera returned offline
with no current metrics, rather than continuing to show the previous count.
These are integration checks; they do not improve the known count errors on 419.

146 software tests passed; one separate opt-in live-stack test skipped. The
independent real-storage smoke check above passed. Ruff and MyPy passed.
Checkpoint weights were not trained or changed. Physical RPi capture remains
untested because the camera is not mounted.

## Free hosting status (October 22)

The default `render.yaml` and `render.free.yaml` create only two Free web
services. The earlier paid configuration is archived as `render.paid.yaml` and
was not deployed. No card was added. The API uses frozen FP32 ONNX models and
SQLite; the dashboard reads the API over its public HTTPS URL. No model training
or dataset-specific threshold changes are part of this deployment.

The free API and dashboard resources were created in Render. The first API
build failed during the Ultralytics import because desktop OpenCV required
`libxcb.so.1` in the slim exporter image. `Dockerfile.free-api` now removes the
desktop OpenCV distribution and installs only headless OpenCV before importing
Ultralytics. The runtime already uses headless OpenCV.

Commit `9c40c71` built and deployed successfully on Render in 2m39s (deploy
`dep-dbctnm6q1p3s73d9iqj0`). The public API `/ready` returned HTTP 200 with
`inference_backend=onnx`, `storage_backend=sqlite`, and `sqlite=true`. The
public dashboard health endpoint returned HTTP 200 and its actual UI loaded
Mess Main with the expected offline message before any fresh uploads.
Unauthenticated ingestion returned HTTP 401, as required.

- API: https://mqm-free-snapshot-api.onrender.com
- Dashboard: https://mqm-free-snapshot-dashboard.onrender.com

Authenticated remote snapshot/count/status/history/dashboard validation remains
pending permission to access the existing Render camera secret. Startup and
health success do not prove completed image inference. Free instances sleep
when idle, share the workspace's included hours, and lose SQLite history when their ephemeral filesystem resets. Physical
RPi capture remains untested. No raw supplied photos, labels, or credentials are
part of the deployment commits.

The latest software suite passed 154 tests with one opt-in live-stack test
skipped. Ruff and MyPy passed before this Docker-only build fix. Local model
exports preserved both source checkpoint hashes. Cloud inference parity and
memory use still require remote verification; the Mac is not hosting inference.
