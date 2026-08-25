"""
inference/eval_band_recall.py

NOT part of the final deliverable. Measures whether detection misses are
concentrated in specific regions of the frame (e.g. far/back of hall) vs
spread uniformly — a scalar count can't tell those apart, and they imply
different fixes downstream (tune globally vs. treat far-zone counts as
unreliable in metrics.py).

Usage:
    1. Fill in GROUND_TRUTH_COUNTS below by hand-counting people in each
       horizontal band of ONE test image (near/mid/far thirds of the
       frame, top-to-bottom or however the camera is actually angled).
    2. Run: python inference/eval_band_recall.py path/to/image.jpg
    3. Compares model output falling in each band against your hand count.

This is a one-image sanity check, not a proper eval set — good enough to
answer "uniformly mediocre vs blind past a certain distance," not good
enough to certify accuracy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.model import detect_people  # noqa: E402

# --- Fill these in by hand-counting the test image ----------------------
# Bands are horizontal strips of the frame, defined as fractions of image
# height. Adjust band boundaries to match where "near/mid/far" actually
# falls in your specific camera angle.
BAND_BOUNDARIES = {
    "near": (0.66, 1.0),   # bottom third of frame = closest to camera
    "mid": (0.33, 0.66),
    "far": (0.0, 0.33),    # top third = farthest from camera
}

# TODO: hand-count people in each band of your test image and fill in.
GROUND_TRUTH_COUNTS = {
    "near": None,  # e.g. 12
    "mid": None,   # e.g. 20
    "far": None,   # e.g. 35
}


def band_for_detection(bbox: list[float], image_height: int) -> str | None:
    """Assigns a detection to a band based on its vertical center."""
    y_center = (bbox[1] + bbox[3]) / 2
    frac = y_center / image_height
    for band, (lo, hi) in BAND_BOUNDARIES.items():
        if lo <= frac < hi:
            return band
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_band_recall.py <image_path>")
        sys.exit(1)

    if any(v is None for v in GROUND_TRUTH_COUNTS.values()):
        print(
            "GROUND_TRUTH_COUNTS not filled in — hand-count people in each "
            "band of the test image first and edit this file."
        )
        sys.exit(1)

    import cv2

    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    height = img.shape[0]

    detections = detect_people(image_path)

    band_detected_counts = {"near": 0, "mid": 0, "far": 0}
    for d in detections:
        band = band_for_detection(d["bbox"], height)
        if band:
            band_detected_counts[band] += 1

    print(f"{'Band':<6} {'Detected':<10} {'Ground truth':<14} {'Count ratio':<12}")
    for band in ("near", "mid", "far"):
        detected = band_detected_counts[band]
        truth = GROUND_TRUTH_COUNTS[band]
        ratio = detected / truth if truth else float("nan")
        print(f"{band:<6} {detected:<10} {truth:<14} {ratio:.0%}")
    print(
        "\nNOTE: this is a COUNT RATIO, not recall. Detections are never matched\n"
        "to specific ground-truth people, so a false positive inside a band\n"
        "inflates the number and a value >100% means false positives, not\n"
        "perfect detection. Use it to spot spatial bias (is the far band much\n"
        "worse than near?), not to certify accuracy."
    )