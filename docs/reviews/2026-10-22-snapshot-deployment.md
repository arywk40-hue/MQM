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

## Hosting configuration

`compose.snapshot.yml` provides API, dashboard, private Redis, and persistent
InfluxDB on one host. `render.yaml` provides the analogous Render Blueprint.
The Blueprint passed Render's published JSON Schema validation. The signed-in
Render dashboard found the file on `codex/04-integration-quality`, but showed
**Payment Information Required** because the account has no payment details for
the selected paid API/history services. No paid service creation or public
deployment was submitted. The card-entry step remains for the account owner.
Public hosting and its recurring resource charge must be verified in Render;
preparing or pushing this configuration does not itself mean the service is live.
No raw supplied photos, labels, or credentials are part of the deployment commit.

The local Linux container build started but is still downloading dependencies
on this connection; the native integration check is not a completed container
image validation. Production URLs and a remote smoke check must be recorded
after deployment before describing the public system as working.
