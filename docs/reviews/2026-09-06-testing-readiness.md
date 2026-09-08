# Local testing readiness — 6 September 2026

The already-running local application is available for manual testing at
http://127.0.0.1:8501, with API documentation at http://127.0.0.1:8000/docs.

## Changes

- `make dev` now waits for `/ready`, including Redis and InfluxDB health,
  before starting Streamlit. Each readiness request has a five-second timeout.
- `make test-live` runs the existing opt-in integration test using `.env`.
- The live test now checks the upload's exact timestamp and all historical
  metric fields, preventing old history from satisfying the test.
- README code fences and live-test instructions were corrected.

## Verification

- Ruff lint and formatting, shell syntax, Compose configuration, and Git
  whitespace checks passed.
- Both local storage containers reported healthy; API `/ready` and Streamlit
  `/_stcore/health` returned successful responses.
- Authenticated upload of `mess_hall_dense.jpg` succeeded against the existing
  API. An immediate current-status read returned the identical metrics and
  timestamp with `online: true`.
- History contained the uploaded reading at `2026-09-06T12:49:03.663338Z`:
  headcount 27, queue 1, occupied seats 18/64, occupancy 28.1%, crowd amber.
- A later status read correctly reported offline after the freshness window.

## Limits

The full pytest suite, strengthened live pytest test, and MyPy did not finish.
Processes blocked reading Python dependencies; macOS reports some `.venv`
files as `dataless`. Retrying with a fresh bytecode-cache directory also stalled
on source-file reads. Restore those files locally or recreate the virtual
environment, then run `make test lint typecheck format-check` and, with the
application running, `make test-live`.

The smoke checks exercised the existing server process, which was not restarted.
The revised startup script was syntax-checked but not exercised by restarting
the user's running application. No new full-suite pass is claimed.

Real mounted-camera testing still requires matching capture resolution and
zone calibration; the committed sample uses 736×490 geometry. Sample uploads
expire from current status unless frames continue arriving.
