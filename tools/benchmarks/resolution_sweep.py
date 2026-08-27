"""
Model x inference-resolution sweep.

Not part of the deployed system. Produces the evidence for two decisions:
which weights to ship, and what imgsz to run them at.

The initial comparison at a single resolution was inconclusive because
detection count had not plateaued at the highest resolution tested. This sweeps
both axes and reports where the return on resolution flattens.

Emits a markdown table to stdout and a CSV alongside it for charting.

    python tools/benchmarks/resolution_sweep.py data/samples/mess_hall_dense.jpg
    python tools/benchmarks/resolution_sweep.py <image> --models yolov8n.pt,yolov8s.pt
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.model import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    resolve_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "outputs"

DEFAULT_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolo11x.pt", "rtdetr-l.pt"]
DEFAULT_SIZES = [640, 960, 1280, 1600, 1920, 2560]
TIMED_RUNS = 3


def measure(model, image_path: str, imgsz: int) -> tuple[int, float]:
    """Detection count and median wall-clock seconds, excluding warmup."""
    model.predict(
        image_path,
        conf=DEFAULT_CONFIDENCE_THRESHOLD,
        iou=DEFAULT_IOU_THRESHOLD,
        imgsz=imgsz,
        classes=[0],
        verbose=False,
    )

    timings, results = [], None
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        results = model.predict(
            image_path,
            conf=DEFAULT_CONFIDENCE_THRESHOLD,
            iou=DEFAULT_IOU_THRESHOLD,
            imgsz=imgsz,
            classes=[0],
            verbose=False,
        )
        timings.append(time.perf_counter() - start)

    boxes = results[0].boxes
    return (len(boxes) if boxes is not None else 0), statistics.median(timings)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    ap.add_argument("--save-annotated", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    from PIL import Image

    with Image.open(args.image) as im:
        src_w, src_h = im.size
    print(f"source: {args.image}  {src_w}x{src_h}")
    print(
        f"conf={DEFAULT_CONFIDENCE_THRESHOLD}  iou={DEFAULT_IOU_THRESHOLD}  "
        f"median of {TIMED_RUNS} timed runs after warmup\n"
    )

    rows: list[dict] = []
    for weights in models:
        model = YOLO(resolve_weights(weights))
        for imgsz in sizes:
            count, secs = measure(model, args.image, imgsz)
            upscale = imgsz / max(src_w, src_h)
            rows.append(
                {
                    "model": weights,
                    "imgsz": imgsz,
                    "count": count,
                    "seconds": round(secs, 3),
                    "upscale_factor": round(upscale, 2),
                }
            )
            print(f"  {weights:<14} {imgsz:>5}  {count:>3} detected  {secs:6.2f}s")

            if args.save_annotated:
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                res = model.predict(
                    args.image,
                    conf=DEFAULT_CONFIDENCE_THRESHOLD,
                    iou=DEFAULT_IOU_THRESHOLD,
                    imgsz=imgsz,
                    classes=[0],
                    verbose=False,
                )
                stem = Path(weights).stem
                Image.fromarray(res[0].plot()[:, :, ::-1]).save(
                    OUT_DIR / f"sweep_{stem}_{imgsz}.jpg"
                )
        print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Timestamped so a narrower follow-up sweep cannot overwrite the full run.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = OUT_DIR / f"resolution_sweep_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(_markdown_table(rows, models, sizes))
    print(f"\ncsv: {csv_path.relative_to(REPO_ROOT)}")
    _report_plateau(rows, models)
    return 0


def _markdown_table(rows: list[dict], models: list[str], sizes: list[int]) -> str:
    by = {(r["model"], r["imgsz"]): r for r in rows}
    out = ["\n### Detections (time in seconds)\n"]
    out.append("| model | " + " | ".join(str(s) for s in sizes) + " |")
    out.append("|---" * (len(sizes) + 1) + "|")
    for m in models:
        cells = []
        for s in sizes:
            r = by.get((m, s))
            cells.append(f"{r['count']} ({r['seconds']:.2f}s)" if r else "-")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def _report_plateau(rows: list[dict], models: list[str]) -> None:
    """Flag models whose detection count is still climbing at the top size."""
    print("\n### Plateau check\n")
    for m in models:
        series = sorted([r for r in rows if r["model"] == m], key=lambda r: r["imgsz"])
        if len(series) < 2:
            continue
        counts = [r["count"] for r in series]
        peak = max(counts)
        peak_at = series[counts.index(peak)]["imgsz"]
        last, prev = series[-1], series[-2]
        delta = last["count"] - prev["count"]

        # A count that falls away from its peak is not a plateau. Transformer
        # detectors with fixed positional embeddings degrade outside their
        # native input size rather than saturating, and reporting that as a
        # plateau would recommend exactly the wrong resolution.
        if peak > 0 and last["count"] < peak * 0.6:
            print(
                f"  {m}: DEGRADES beyond {peak_at} — peak {peak}, "
                f"down to {last['count']} at {last['imgsz']}. Not resolution-"
                f"scalable; evaluate only at {peak_at}."
            )
        elif delta > 0:
            print(
                f"  {m}: still climbing at {last['imgsz']} "
                f"(+{delta} from {prev['imgsz']}). True optimum is higher; "
                f"extend the sweep or raise capture resolution."
            )
        else:
            print(f"  {m}: plateaued by {last['imgsz']} ({delta:+d}), peak {peak} at {peak_at}.")


if __name__ == "__main__":
    sys.exit(main())
