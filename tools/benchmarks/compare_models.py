"""
inference/compare_models.py

NOT part of the final deliverable — a throwaway comparison script for
sanity-checking model choice + inference params on real test photos
before locking in defaults in model.py.

Fixes applied per review (see PR notes):
  - imgsz now passed through to .predict() instead of silently defaulting
    to 640. Previously every model in this comparison ran at 640 while
    model.py runs at 1280 — the comparison wasn't describing the config
    we actually deploy.
  - CONF_THRESHOLD imported directly from model.py's
    DEFAULT_CONFIDENCE_THRESHOLD instead of a separately hardcoded
    number, so this script can't silently drift out of sync with the
    real default again.
  - Added an imgsz sweep (640/960/1280/1920) on a single chosen model.
  - Added augment=True test (already a supported param in model.py,
    previously unused here).
  - Added wall-clock inference time per run — no cost data existed
    before this, and it matters given a free-tier deploy.

Usage:
    python inference/compare_models.py path/to/test_image.jpg
"""

import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.model import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    resolve_weights,
)

CONF_THRESHOLD = DEFAULT_CONFIDENCE_THRESHOLD  # kept in sync with model.py, not duplicated
IOU_THRESHOLD = DEFAULT_IOU_THRESHOLD
IMAGE_SIZE = DEFAULT_IMAGE_SIZE


def run_model(
    weights: str, image_path: str, name: str, imgsz: int = IMAGE_SIZE, augment: bool = False
):
    model = YOLO(resolve_weights(weights))

    # Warm up: first predict() pays graph init / lazy alloc. Production runs a
    # warm singleton via get_detector(), so timing a cold call overstates real
    # per-frame cost. Discard the first run, then time the median of 3.
    model.predict(
        image_path, conf=CONF_THRESHOLD, imgsz=imgsz, augment=augment, classes=[0], verbose=False
    )

    timings = []
    for _ in range(3):
        start = time.perf_counter()
        results = model.predict(
            image_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=imgsz,
            augment=augment,
            classes=[0],
            verbose=False,
        )
        timings.append(time.perf_counter() - start)
    elapsed = sorted(timings)[1]  # median of 3

    n = len(results[0].boxes) if results[0].boxes is not None else 0
    tag = f"{name} @ imgsz={imgsz}" + (" +augment" if augment else "")
    print(f"{tag}: {n} people detected  ({elapsed:.2f}s)")

    annotated = results[0].plot()
    out_dir = Path(__file__).resolve().parents[2] / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"annotated_{name.lower().replace(' ', '_').replace('-', '_')}_{imgsz}.jpg"
    cv2.imwrite(str(out_path), annotated)
    return n, elapsed


def run_sahi(image_path: str, slice_size: int = 320):
    """
    SAHI (Slicing Aided Hyper Inference): splits the image into overlapping
    tiles, runs detection on each tile separately, then merges results.
    Effectively increases resolution-per-person for small/distant people
    without needing a bigger model — this is why it won the resolution
    comparison, not because YOLOv8n got smarter.

    NOTE: AutoDetectionModel has no classes=[0] filter — it runs full
    80-class COCO on every tile. Must filter object_prediction_list to
    person BEFORE both counting and exporting visuals, or the exported
    image shows chairs/tables/etc alongside people. This bit us once
    already — don't reintroduce it.
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=resolve_weights("yolov8n.pt"),
        confidence_threshold=CONF_THRESHOLD,
        device="cpu",
    )

    start = time.perf_counter()
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    elapsed = time.perf_counter() - start

    # Filter to person-only IN PLACE before exporting — export_visuals()
    # draws whatever is in object_prediction_list.
    result.object_prediction_list = [
        p for p in result.object_prediction_list if p.category.name == "person"
    ]

    n = len(result.object_prediction_list)
    print(f"SAHI (tiled, slice={slice_size}) + YOLOv8n: {n} people detected  ({elapsed:.2f}s)")
    out_dir = Path(__file__).resolve().parents[2] / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.export_visuals(export_dir=str(out_dir), file_name=f"annotated_sahi_{slice_size}")
    return n, elapsed


def imgsz_sweep(weights: str, image_path: str, name: str):
    """Sweep imgsz on a single model to see where detection count plateaus."""
    print(f"\n--- imgsz sweep: {name} ---")
    for imgsz in (640, 960, 1280, 1920):
        run_model(weights, image_path, name, imgsz=imgsz)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_models.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print(
        f"Running all models at imgsz={IMAGE_SIZE}, conf={CONF_THRESHOLD} "
        "(matches model.py defaults)\n"
    )

    # 1. YOLOv8n — current default in model.py
    run_model("yolov8n.pt", image_path, "YOLOv8n")

    # 2. YOLOv8n + TTA — augment=True is already a supported param in
    #    model.py but was never actually exercised in this comparison.
    #    Affordable at a ~10s capture cadence; check if it moves the count.
    run_model("yolov8n.pt", image_path, "YOLOv8n", augment=True)

    # 3. YOLOv8s — bigger YOLO variant, same family
    run_model("yolov8s.pt", image_path, "YOLOv8s")

    # 4. YOLO11x — largest single-pass model available
    run_model("yolo11x.pt", image_path, "YOLO11x")

    # 5. RT-DETR — transformer-based, architecturally different from YOLO
    run_model("rtdetr-l.pt", image_path, "RT-DETR")

    # 6. SAHI tiled inference — previously won by a wide margin because
    #    tiling increases effective resolution per person; expect the
    #    gap to narrow once every model runs at 1280 instead of 640.
    try:
        for slice_size in (320, 480, 640):
            run_sahi(image_path, slice_size=slice_size)
    except ImportError:
        print("SAHI not installed — run: pip install sahi")

    # 7. imgsz sweep on whichever model comes out ahead above. Hardcoded
    #    to YOLOv8s here as a starting guess — change to whatever wins.
    imgsz_sweep("yolov8s.pt", image_path, "YOLOv8s")

    print(
        "\nCompare the saved annotated_*.jpg files side by side.\n"
        "Per review: don't change model.py's DEFAULT_MODEL_WEIGHTS until "
        "this corrected run actually justifies it."
    )
