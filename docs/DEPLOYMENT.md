# Deployment

The API and dashboard share one container image and the same model checkpoints.
`compose.deploy.yml` runs them as separate processes with restart policies and
health checks. No raw training/evaluation photos are copied into the image.

## Single host with Docker Compose

1. Install Docker Engine/Compose on the server.
2. Set backend credentials in `.env` using `.env.example`. Use managed Redis
   and InfluxDB, or a separately secured storage stack. Container `localhost`
   refers to the container itself, not the host; use reachable service URLs.
3. Build and start:

   ```bash
   docker compose -f compose.deploy.yml up -d --build
   docker compose -f compose.deploy.yml ps
   curl --fail http://127.0.0.1:8000/ready
   curl --fail http://127.0.0.1:8501/_stcore/health
   ```

4. Route an HTTPS reverse proxy to the loopback dashboard port 8501, with
   WebSocket support. Route authenticated camera ingestion to API port 8000.
   Keep database ports private. The dashboard receives only the read API URL;
   backend credentials are passed only to the API.
5. Calibrate `config/zones.json` for the mounted camera. Current sample geometry
   does not apply to arbitrary real images. Photo/live distance queue and posture estimates do not need zones.

The phone video mode currently uses local peer connectivity without STUN/TURN.
Remote internet video requires separately configured WebRTC connectivity and
HTTPS; uploaded photos and API ingestion do not depend on that mode.

The container uses CPU PyTorch and runs as a non-root user. First model loading
can be slow. `/ready` checks storage availability; it does not measure model
accuracy. Start with at least 4 GB RAM and measure latency on the actual host.

## Evaluation before rollout

```bash
python -m pip install -r requirements-eval.txt
python -m tools.evaluation.real_images \
  --images /path/to/local/photos --labels /path/to/dataset_labels.json \
  --adaptive-confidence
```

Reports are written under ignored `data/outputs/real-images/`. HEIC input is
supported by the evaluation extra. JPEG/PNG are supported in dashboard uploads.
The checked-in aggregate result is in `docs/evaluation/2026-10-21-summary.json`.
The current count errors are significant in dense scenes; treat output as an
estimate and validate on a separate held-out capture session before operational use.

This configuration has been parsed locally. The image has not been built here:
Docker Desktop's daemon was unavailable. A public deployment still requires the
hosting target and credentials.

For fixed-cutoff comparisons, omit `--adaptive-confidence`. Runtime uses the crowd
guard by default; `MODEL_ADAPTIVE_CONFIDENCE=false` disables it. The default 0.25
camera-zone floor continues to apply to calibrated API metrics until camera-specific
validation supports changing it.
