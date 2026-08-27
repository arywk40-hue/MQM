"""Small dependency-free load probe for the public read endpoint."""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def request_once(url: str, timeout: float) -> tuple[int, float]:
    started = time.perf_counter()
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code, time.perf_counter() - started
    except requests.RequestException:
        return 0, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent GET load probe")
    parser.add_argument("--url", default="http://127.0.0.1:8000/status")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(request_once, args.url, args.timeout) for _ in range(args.requests)
        ]
        results = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - started

    statuses: dict[int, int] = {}
    for status, _ in results:
        statuses[status] = statuses.get(status, 0) + 1
    latencies = sorted(latency for _, latency in results)
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    print(f"requests={len(results)} concurrency={args.concurrency} elapsed={elapsed:.2f}s")
    print(
        f"throughput={len(results) / elapsed:.1f} req/s "
        f"median={statistics.median(latencies):.3f}s p95={p95:.3f}s"
    )
    print(f"statuses={statuses}")
    return 0 if statuses == {200: len(results)} else 1


if __name__ == "__main__":
    raise SystemExit(main())
