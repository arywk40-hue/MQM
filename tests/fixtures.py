"""
tests/fixtures.py

Synthetic detections in inference/model.py's output format, so the zone and
metrics layers can be built and tested before the detector, the real camera,
or ultralytics exist on the machine.

Everything here is in the 736x490 coordinate space of config/zones.json's
`mess_main` placeholder geometry. When the real camera frame lands and those
polygons are retraced, these fixtures need retracing too.
"""

from __future__ import annotations


def det(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> dict:
    """One detection in model.py's output format."""
    return {"bbox": [x1, y1, x2, y2], "confidence": conf}


def person_at(cx: float, feet_y: float, height: float = 60.0, conf: float = 0.9) -> dict:
    """
    A person standing at (cx, feet_y).

    Built from the floor-contact point outward, because that is what the zone
    layer actually tests against — makes the intent of each fixture explicit
    rather than leaving it implied by four corner numbers.
    """
    width = height * 0.4
    return det(cx - width / 2, feet_y - height, cx + width / 2, feet_y, conf)


# --- Scenarios ---------------------------------------------------------
# Coordinates chosen to sit well inside the placeholder polygons in
# config/zones.json, not on their edges.

EMPTY: list[dict] = []

# 3 people in the queue polygon [[286,148],[430,143],[446,196],[281,200]]
QUEUE_ONLY = [
    person_at(320, 180, height=30),
    person_at(360, 178, height=30),
    person_at(400, 176, height=30),
]

# 4 people in seating_front [[24,268],[736,246],[736,490],[24,490]]
SEATED_ONLY = [
    person_at(200, 400, height=90),
    person_at(350, 420, height=90),
    person_at(500, 380, height=90),
    person_at(650, 440, height=90),
]

# Lunch rush: queue + both seating bands + entrance
BUSY = (
    QUEUE_ONLY
    + SEATED_ONLY
    + [
        person_at(120, 230, height=45),  # seating_mid
        person_at(300, 240, height=45),  # seating_mid
        person_at(480, 235, height=45),  # seating_mid
        person_at(600, 225, height=45),  # seating_mid
        person_at(50, 220, height=50),  # entrance
    ]
)

# Below the 0.25 min_confidence floor in config/zones.json — should be dropped
LOW_CONFIDENCE = [
    person_at(320, 180, height=30, conf=0.05),
    person_at(360, 178, height=30, conf=0.10),
]

# Nobody inside any polygon (top strip of the frame, above every zone)
OUTSIDE_ALL_ZONES = [
    person_at(400, 40, height=20),
    person_at(500, 60, height=20),
]
