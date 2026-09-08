#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
venv_dir="${MQM_VENV_DIR:-$repo_root/.venv}"

if [[ ! -f .env ]]; then
  echo "Missing .env. Run: cp .env.example .env, then replace every change-me value." >&2
  exit 1
fi
if [[ ! -x "$venv_dir/bin/python" ]]; then
  echo "Missing Python environment at $venv_dir. Run make install or set MQM_VENV_DIR." >&2
  exit 1
fi

docker compose up -d --wait

cleanup() {
  kill "${api_pid:-}" "${dashboard_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$venv_dir/bin/python" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
api_pid=$!

# Phase 1: wait for process to bind
for _ in {1..30}; do
  if curl --silent --fail --max-time 2 http://127.0.0.1:8000/health > /dev/null; then
    break
  fi
  if ! kill -0 "$api_pid" 2>/dev/null; then
    echo "API exited during startup." >&2
    exit 1
  fi
  sleep 1
done
if ! curl --silent --fail --max-time 2 http://127.0.0.1:8000/health > /dev/null; then
  echo "API did not start within 30 seconds." >&2
  exit 1
fi
# Phase 2: confirm dependencies (Redis + InfluxDB) are connected
for _ in {1..15}; do
  if curl --silent --fail --max-time 5 http://127.0.0.1:8000/ready > /dev/null; then
    break
  fi
  sleep 1
done
if ! curl --silent --fail --max-time 5 http://127.0.0.1:8000/ready > /dev/null; then
  echo "API dependencies did not become ready. Check Redis/InfluxDB configuration." >&2
  exit 1
fi

PYTHONPATH="$repo_root" "$venv_dir/bin/python" -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501 &
dashboard_pid=$!

echo "API:       http://127.0.0.1:8000/docs"
echo "Dashboard: http://127.0.0.1:8501"
echo "Press Ctrl-C to stop the API and dashboard (Redis/InfluxDB remain running)."

while kill -0 "$api_pid" 2>/dev/null && kill -0 "$dashboard_pid" 2>/dev/null; do
  sleep 1
done
echo "A development process exited unexpectedly." >&2
exit 1
