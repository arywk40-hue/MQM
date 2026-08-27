"""
tools/draw_zones.py

Renders config/zones.json over a frame so polygons can be eyeballed before
trusting any metric. Uses PIL only — no ultralytics, no OpenCV.

    python tools/draw_zones.py Human_detection/b.jpg mess_main
    python tools/draw_zones.py frame.jpg mess_main --out check.png

Also prints the frame's actual dimensions against what the config declares.
That mismatch is the failure mode that makes every metric silently read zero,
so it is worth seeing before anything else.

Tracing a new camera:
    1. Grab one frame at the camera's real capture resolution.
    2. Open it in any image editor, hover the corners of each region, note
       the pixel coordinates.
    3. Put them in config/zones.json, set frame_width/frame_height to the
       frame's real size.
    4. Run this and check the overlay lands where you meant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.zones import load_zones  # noqa: E402

ZONE_COLOURS = {
    "queue": (255, 92, 92),
    "seating": (92, 176, 255),
    "entrance": (255, 200, 64),
}
FALLBACK_COLOUR = (170, 170, 170)


def main() -> int:
    ap = argparse.ArgumentParser(description="Overlay zone polygons on a frame.")
    ap.add_argument("image")
    ap.add_argument("camera_id")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="default: <image stem>_zones.png")
    args = ap.parse_args()

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow required: pip install pillow")
        return 1

    config_path = args.config or (Path(__file__).resolve().parent.parent / "config" / "zones.json")
    config = load_zones(args.camera_id, config_path)

    img = Image.open(args.image).convert("RGBA")
    print(f"frame:  {img.width}x{img.height}   ({args.image})")
    print(f"config: {config.frame_width}x{config.frame_height}   (camera {args.camera_id!r})")

    if (img.width, img.height) != config.frame_size:
        print(
            "\nMISMATCH. The polygons were traced against a different resolution "
            "than this frame.\nIn production this raises rather than silently "
            "scoring every zone check false.\nRetrace the polygons, or use "
            "zones.rescale() if it is the same scene at a new size.\n"
            "Drawing anyway so you can see how far off it is."
        )

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for zone in config.zones:
        colour = ZONE_COLOURS.get(zone.type, FALLBACK_COLOUR)
        pts = [tuple(p) for p in zone.polygon]
        draw.polygon(pts, fill=colour + (60,), outline=colour + (255,))
        for x, y in pts:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=colour + (255,))

        label = zone.name if zone.seats is None else f"{zone.name} ({zone.seats} seats)"
        lx = min(p[0] for p in pts) + 4
        ly = min(p[1] for p in pts) + 2
        draw.text((lx + 1, ly + 1), label, fill=(0, 0, 0, 200))
        draw.text((lx, ly), label, fill=colour + (255,))

    out = args.out or str(Path(args.image).with_name(Path(args.image).stem + "_zones.png"))
    Image.alpha_composite(img, overlay).convert("RGB").save(out)
    print(f"\nwrote {out}")

    print("\nzones:")
    for zone in config.zones:
        seats = "" if zone.seats is None else f", {zone.seats} seats"
        print(f"  {zone.name:<16} {zone.type:<10} {len(zone.polygon)} pts{seats}")
    print(f"  total seats: {config.total_seats}")
    print(f"  crowd bands: {config.crowd_thresholds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
