# Model selection: measured results

First run of the corrected benchmark tooling. Supersedes the conclusions in
`2026-08-24-model-review.md`, which were drawn from a comparison that ran every
model at 640 while production runs 1280.

Test image: `data/samples/mess_hall_dense.jpg`, 736x490, roughly 80-120 people.
Hardware: local CPU, torch 2.13, no GPU. `conf=0.25`, `iou=0.5`, median of three
timed runs after a discarded warmup.

Raw data: `data/outputs/resolution_sweep.csv`, chart in
`data/outputs/resolution_sweep.png`.

## Detections by model and resolution

| model | 640 | 960 | 1280 | 1600 | 1920 | 2560 | 3200 | 4096 |
|---|---|---|---|---|---|---|---|---|
| yolov8n | 8 | 19 | 27 | 34 | 41 | **43** | 40 | 30 |
| yolov8s | 16 | 22 | 24 | 34 | 36 | **49** | 42 | 31 |
| yolo11x | 5 | 12 | 31 | 43 | 55 | **72** | 71 | 59 |
| rtdetr-l | **27** | 29 | 29 | 16 | 5 | 0 | - | - |

Seconds per frame:

| model | 640 | 1280 | 1920 | 2560 | 4096 |
|---|---|---|---|---|---|
| yolov8n | 0.06 | 0.22 | 0.57 | 1.12 | 2.78 |
| yolov8s | 0.13 | 0.52 | 1.22 | 2.66 | 6.66 |
| yolo11x | 0.84 | 3.38 | 7.37 | 14.55 | 37.23 |
| rtdetr-l | 0.79 | 3.22 | 7.74 | 11.19 | - |

## Findings

### 1. Resolution dominates architecture

YOLOv8n moves from 8 detections to 43 across the resolution range — a factor of
5.4 on unchanged weights. At any fixed resolution the spread between the
smallest and largest model is far narrower than that.

The round-1 hypothesis is confirmed, and more strongly than expected. Choosing
weights before choosing an inference resolution optimises the wrong variable.

### 2. Every YOLO model peaks at 2560, then declines

All three peak at exactly 2560 and fall away above it. 2560 is a 3.5x upscale of
the 736px source, which suggests the models are not gaining information — they
are being fed subjects at the pixel size their anchor scales were trained for.
Beyond that, upscaling blurs faster than it enlarges.

This number is a property of *this* source resolution, not a universal setting.
A 1080p or 4K capture will peak somewhere else entirely. Do not hardcode 2560.

### 3. RT-DETR is not resolution-scalable

27 → 29 → 29 → 16 → 5 → **0**.

The transformer's positional embeddings are fixed to its native 640 input, so
it degrades to complete failure rather than saturating. At 640 it is the
strongest model tested — 27 detections against YOLOv8n's 8 — but it cannot
exploit the resolution headroom that turns out to be the deciding factor.

Consequence: RT-DETR is only meaningful at 640, and at 640 it is beaten by
YOLOv8n at 2560 (43) for a fifth of the compute. Not a candidate.

The earlier round-1 note that "RT-DETR and YOLO roughly agreeing means the count
is trustworthy" does not hold: they agree at 1280 by coincidence, on opposite
sides of their respective curves.

### 4. SAHI is obsolete

| approach | detections | seconds |
|---|---|---|
| SAHI, 320px slices | 30 | 1.24 |
| SAHI, 480px slices | 20 | 0.39 |
| SAHI, 640px slices | 14 | 0.25 |
| **YOLOv8n at 2560** | **43** | **1.12** |

SAHI won the original comparison only because the baseline was crippled at 640.
Tiling and upscaling both work by increasing pixels-per-subject; upscaling does
it more cheaply here. Plain inference beats the best SAHI configuration on both
axes simultaneously.

Recommend dropping the SAHI dependency. It adds an install, a code path, and a
class-filtering footgun for no benefit.

### 5. YOLO11x is accurate and unaffordable

72 detections at 2560, comfortably the best. It costs 14.55s per frame on local
CPU against a 10-second capture cadence — already over budget before accounting
for free-tier hardware being slower than this machine.

Detections per second of compute:

| model | detections | s/frame | per second |
|---|---|---|---|
| yolov8n @ 2560 | 43 | 1.12 | 38.4 |
| yolov8s @ 2560 | 49 | 2.66 | 18.4 |
| yolo11x @ 2560 | 72 | 14.55 | 4.9 |

YOLO11x buys 67% more detections for 13x the compute.

### 6. False positives confirmed, and they are static

`person 0.31` and `person 0.44` in `data/outputs/best_yolo11x_2560.jpg` are
boxes drawn on the painted pillars. The murals contain human figures and the
detector reads them as people.

This matters disproportionately for a fixed camera. A mural does not move, so it
contributes a constant offset to headcount in every frame, indefinitely.
Crowd-level thresholds calibrated against a biased baseline will inherit the
bias silently.

Two mitigations, both in the zone layer rather than the detector:

- Exclude known-static false positives geometrically, via a masked zone.
- Flag detections whose bounding box is identical across consecutive frames.
  Real diners move between 10-second captures; a mural does not.

Worth revisiting once the real camera is mounted, since it depends entirely on
what is in frame.

### 7. Confidence threshold is the largest untuned lever

Detections retained by YOLO11x at 2560 as the floor rises:

| conf | 0.05 | 0.15 | 0.25 | 0.35 | 0.50 | 0.70 |
|---|---|---|---|---|---|---|
| detections | 136 | 95 | 72 | 58 | 47 | 26 |

A 5x swing from one parameter, larger than the effect of model choice. The
round-2 disagreement over 0.2 versus 0.25 was arguing about the wrong end of a
range this wide, and neither value is grounded in anything measured.

This cannot be settled without ground truth. At `conf=0.05` the model reports 136
subjects in a hall that plausibly holds 80-120, so some of that tail is certainly
false. Which portion is unknown.

## Recommendation

**Provisionally: YOLOv8n at imgsz=2560.** Best detections per unit compute, no
weights change required (`DEFAULT_MODEL_WEIGHTS` already points at it), and
comfortably inside the cadence budget with room for slower deployment hardware.

The change this implies is `DEFAULT_IMAGE_SIZE` from 1280 to 2560 — but hold it
until validated on real footage, because 2560 is calibrated to a 736px source
and will be wrong for a different capture resolution.

If deployment hardware turns out to have GPU headroom, YOLOv8s at 2560 is the
next step up: 49 detections at 2.66s.

## What this does not establish

Every number here is a **detection count, not an accuracy measurement**. Nothing
was matched against ground truth, so "43 detections" could be 43 correct, or 38
correct and 5 murals. Finding 6 proves the second category is non-empty.

Two consequences:

- The ranking is trustworthy insofar as all models were measured identically on
  the same image. The absolute figures are not.
- Any threshold calibrated on these numbers inherits their error.

Outstanding to close this properly:

1. Ground-truth count on one image, so `tools/benchmarks/eval_band_recall.py`
   can run. Hand-counting 80-120 overlapping subjects at this resolution is
   unreliable; a higher-resolution frame or a cropped region would give a
   defensible number.
2. Per-band recall, to establish whether misses concentrate at the back of the
   hall. This directly determines whether queue count is trustworthy, since the
   serving counter sits at the rear of the frame.
3. Re-run on real mounted-camera footage. Everything above is measured on a
   sourced stock photo at an arbitrary angle.

## Action for the wider team

**Specify the camera capture resolution.** This is now the highest-leverage open
decision on the project. The single largest effect measured — a 5.4x swing —
comes from pixels-per-subject, and capture resolution sets the ceiling on that.
Inference resolution can only upscale what the sensor recorded.

A camera capturing 1080p or better would likely shift the optimum configuration
and could make a smaller, cheaper model sufficient.
