# Mess-dashboard handover integration

## Scope reviewed

The untracked `1/` handover contains two copies of the same API prototype:
`mess-dashboard/` and `mess-dashboard 2/`.  It is an ingestion/read backend,
not a separate dashboard UI.  Its only inference implementation returns
random metrics, so it must not enter the production request path.

## Integration map

| Handover component | Tracked production component | Result |
| --- | --- | --- |
| `api/ingestion.py` | `api/ingestion.py` | Integrated architecture: image upload, inference, history write, latest-state write. Production additionally requires `X-Camera-Key`, accepts JPEG only, checks upload size, and reports a failed durable write as HTTP 503. |
| `api/read.py` | `api/read.py` | Integrated architecture: read-only current-status and history endpoints. Production returns every configured camera with an explicit offline state and returns store failures as HTTP 503. |
| `api/storage/redis_client.py` | `api/storage/redis_client.py` | Integrated latest-reading TTL contract. Production uses one `MGET` for configured cameras instead of scanning Redis. |
| `api/storage/influx_client.py` | `api/storage/influx_client.py` | Integrated timestamped history contract. The checked-in local stack uses InfluxDB 2.x/Flux, not the handover's InfluxDB 3/SQL client. |
| `inference/metrics.py` | `inference/pipeline.py`, `inference/metrics.py`, `inference/zones.py` | Replaced the deliberately random stub with the real JPEG-to-YOLO-to-zones pipeline. |
| endpoint smoke test | `tests/test_api.py`, `tests/test_live_stack.py` | Covered as deterministic unit/API tests plus an opt-in live Redis/InfluxDB round trip. |

## Deliberate non-imports

- Do not copy the duplicate project folders into the tracked tree. Duplicate
  `api/` packages would make imports ambiguous and would drift from the
  production implementation.
- Do not import the random `compute_metrics()` stub. It would hide inference
  failures and violate the real-data requirement.
- Do not retain wildcard CORS or the prototype's behavior of logging an
  InfluxDB write failure and returning success. The tracked app has
  environment-controlled CORS and rejects an ingest unless both storage writes
  succeed.
- Do not add the archived 640x480 sample to the live ingest flow. The active
  `mess_main` zones were traced for 736x490; accepting a mismatched frame would
  produce invalid occupancy and queue metrics.

## InfluxDB 3 decision

The handover describes InfluxDB Cloud Serverless/InfluxDB 3, whereas the
checked-in Compose stack and adapter are InfluxDB 2.x.  Both can support the
same API contract, but their client libraries, query languages, bootstrap
steps, and local ports differ.  A migration must be made only after the
deployment account is confirmed as InfluxDB 3; it should be a separately
tested provider change, not an unverified dual-client fallback.

## Verification

`tests/test_api.py` exercises the handover's intended flow: authenticated
ingest, current status, explicit offline status, history, malformed input,
unknown camera, inference outage, and storage outage.  The full test suite is
run before this handover note is committed.
