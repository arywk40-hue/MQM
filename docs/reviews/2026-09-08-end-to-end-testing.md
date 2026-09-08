# End-to-end testing — 8 September 2026

## Follow-up: slow analysis resolved in the running environment

The running dashboard was blocked reading Torch bytecode in the repository's
`.venv`, before model inference. Using the existing fully local Python 3.11
environment at `/tmp/mqm-photo-verify` completed sample photo analysis in
**3.68 seconds cold and 0.23 seconds warm**, detecting 27 people in both runs.
These are direct photo-analysis timings for the committed sample, not guarantees
for every image or browser interaction.

The API and dashboard were restarted with that environment. `scripts/dev.sh`
now accepts `MQM_VENV_DIR` to select a working environment. The original `.venv`
was preserved; `/tmp` environments are temporary, so this is an operational
workaround rather than a repair of those original dependency files.

Verification with the working environment:

- Full automated suite: **72 passed, 1 opt-in live test skipped** (9.57 seconds).
- Live storage test after restart: **1 passed** (0.45 seconds).
- Real Streamlit AppTest workflow: photo selection, actual YOLO inference,
  annotated PNG validity, download-button presence, result retention across
  reruns, saved-layout classifiers, and actual API metrics/history chart passed.
- MyPy: passed for all 21 source files after two small upload-result typing fixes.
- Focused dashboard/photo regression tests after those fixes: **12 passed**.
- Ruff lint and formatting passed; the previously reported formatting issue was
  corrected. `pip check` found no broken requirements. Launcher shell syntax passed.

Docker Compose recreated the InfluxDB container during restart and retained its
named data volume; subsequent readiness and ingestion/history checks passed.
The tests wrote sample readings. Temporary photo fixtures were removed.
Browser-level file-picker/download interaction and real-camera detection accuracy
remain outside this verification. The original findings below record the earlier run.

## Result

The running local backend passed real ingestion, persistence, readback, invalid-input checks, and a read-endpoint load probe. This is **not a complete end-to-end pass**: formatting failed, and local dependency-file reads blocked the full automated suite and dashboard interaction test.

Tested the working tree including the user's existing staged photo-upload changes. No application code or staged changes were modified. The existing API and dashboard processes were reused; their loaded source revision was not independently established.

## Passed

- `make test-live`: **1 passed in 1.02 seconds**. Uploaded the committed camera sample through the running API and checked Redis current status and timestamp-matched InfluxDB history, including metric equality.
- `python tests/test_zones_metrics.py`: **28/28 passed** using `.venv/bin/python`.
- Ruff lint: all checks passed.
- `docker compose config --quiet`: passed.
- `bash -n scripts/dev.sh`: passed.
- HTTP 200 from API `/health`, API `/ready`, Streamlit `/_stcore/health`, and the dashboard root page.
- Live negative cases: missing/invalid camera keys returned 401; unknown status camera returned 404; invalid history interval returned 422; corrupt JPEG returned 422; PNG ingestion returned 415; wrong image resolution returned 422.
- Repository load tool: **1,000/1,000 HTTP 200**, concurrency 100, elapsed 1.43 seconds, throughput 700.2 requests/second, median latency 0.088 seconds, p95 0.493 seconds. This was a local `/status` read probe, not an inference-throughput benchmark.

## Failed check

`ruff format --check api dashboard inference tests tools` reported **1 file requiring formatting, 38 already formatted**. In `dashboard/app.py`, the local-image filtering condition around line 34 needs Ruff's multiline formatting.

## Blocked verification

- Full `make test` / pytest suite: no completed suite result.
- MyPy type checking: no completed result.
- `pip check`: no completed result.
- Streamlit AppTest interaction with actual YOLO, annotated-image download, saved-layout classifiers, and live metrics/history: HTTP smoke checks completed, but the interaction test did not complete.

Process sampling showed pytest blocked in a file read. Open-file inspection located stalls in `.venv` dependencies, including httpx2 bytecode, Streamlit source, Torch bytecode, and pip bytecode. Some dependency source files carry macOS `dataless` flags. Retrying outside the sandbox and with a separate Python bytecode cache did not resolve the stalls; the latter also blocked reading dependency source. The precise filesystem cause was not established.

The initial sandboxed live test failed because localhost networking was denied. Its retry with local network access passed; that initial failure is an environment restriction, not an API regression.

## Side effects and follow-up

The successful live-stack test wrote a sample reading to the configured local databases. Temporary dashboard sample data was removed, and stalled test processes started for this session were stopped. Existing API, Streamlit, and database services were left running.

To complete verification, make the virtual environment's files fully available locally (or recreate the environment on a fully local filesystem), then rerun the full suite, MyPy, dependency check, and real dashboard interaction test. Apply Ruff formatting to `dashboard/app.py` and rerun the formatting check. Browser-level upload/download interaction and detection accuracy against labeled real-camera footage remain unverified.
