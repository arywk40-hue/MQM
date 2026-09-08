"""Analyze uploaded photos locally without publishing camera readings."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_photo(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Choose a JPEG or PNG image smaller than 10 MB.")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG and PNG images are supported.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Choose an image with no more than 20 million pixels.")
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(
            "This file could not be opened as an image. Try another JPEG or PNG."
        ) from exc


def analyze_photo(data: bytes, camera_id: str | None = None) -> dict:
    image = decode_photo(data)
    frame = np.asarray(image)[:, :, ::-1].copy()
    if camera_id:
        from inference.zones import assign_zones, load_zones

        config = load_zones(camera_id)
        # Validate geometry before running the expensive detector.
        assign_zones([], config, frame_size=image.size)

    from inference.model import detect_people

    detections = detect_people(frame)
    metrics = None
    if camera_id:
        from inference.classifiers import attribute_classifiers_enabled, classify_people
        from inference.metrics import metrics_from_assignments

        assignments = assign_zones(detections, config, frame_size=image.size)
        attributes = (
            classify_people(frame, assignments) if attribute_classifiers_enabled() else None
        )
        metrics = metrics_from_assignments(
            assignments,
            config,
            detections_raw=len(detections),
            queue_flags=[item.is_queue for item in attributes] if attributes else None,
            seated_flags=[item.is_seated for item in attributes] if attributes else None,
        )

    drawing = ImageDraw.Draw(image)
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["bbox"]
        drawing.rectangle((x1, y1, x2, y2), outline="#16a34a", width=3)
        drawing.text(
            (x1 + 3, y1 + 3), str(index), fill="white", stroke_width=2, stroke_fill="black"
        )
    output = BytesIO()
    image.save(output, format="PNG")
    return {
        "headcount": len(detections),
        "detections": detections,
        "metrics": metrics,
        "image": output.getvalue(),
        "width": image.width,
        "height": image.height,
    }
