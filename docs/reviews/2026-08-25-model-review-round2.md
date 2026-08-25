# Review notes round 2: Updated/

Reviewed `model.py`, `model_rtdetr_sahi_compare.py`, `eval_band_recall.py`.

Docstring items are done and correct. Script fixes are in. But nothing has been
run, so the model question is still open.

## Status against round 1 checklist

| # | Item | Status |
|---|---|---|
| 1 | `imgsz` param, conf sync, re-run at 1280 | script fixed, **not run** |
| 2 | imgsz sweep 640/960/1280/1920 | added, not run |
| 3 | Test `augment=True` | added, not run |
| 4 | Per-model inference time | added, measurement flawed (see 3 below) |
| 5 | Per-band recall | script added, GT not filled, metric mislabeled (see 5) |
| 6 | Loosen centroid docstring | done |
| 7 | Document `[]`-on-empty + coordinate space | done |
| 8 | 2-3 images with visible queue | **not done** |
| 9 | Don't change `DEFAULT_MODEL_WEIGHTS` | unchanged, but conf changed instead (see 1) |

Importing `CONF_THRESHOLD` from `model.py` rather than duplicating the literal is
better than what round 1 asked for. Constants can't drift apart again.

No new `annotated_*_1280.jpg` files exist, no sweep output, no timing numbers. We
have corrected instrumentation and no measurements. Please run it.

## 1. Confidence threshold moved in production

`DEFAULT_CONFIDENCE_THRESHOLD` went 0.2 -> 0.25 in `model.py`.

Round 1 asked for the *comparison script* to move to 0.2 to match production. This
synced them in the opposite direction, changing live detector behaviour under a
"keep the constants in sync" change. A higher floor drops exactly the low-confidence
distant detections that are the current failure mode.

Either direction is arguable, but it needs to be a deliberate call with a number
behind it, not a side effect. Revert to 0.2 for now and let the corrected comparison
decide. If 0.25 is the right value, the sweep should show it.

Separately, the inline comment says "tuned against real mess-hall test photo". There
is no real mess-hall footage yet — `b.jpg` is a found image. Please reword; that
comment will mislead whoever tunes this after the camera is mounted.

## 2. b.jpg is 736x490, which caps the resolution experiment

Running a 736x490 source at `imgsz=1280` or `1920` upscales beyond native resolution.
No detail is added by that, so the sweep can't cleanly test whether resolution is the
bottleneck.

It won't be a flat no-op — YOLO detects at fixed anchor scales, so upscaling a 15x30px
person to 26x52 can push it over the detectable threshold. Expect some gain, but read
it as "the network prefers larger input", not "more real detail helped". Round 1
overstated what this sweep can prove; the source dimensions should have been checked
first.

The actionable version: the constraint is **capture** resolution, not inference
resolution, and nobody has specified what the mounted camera records at. Worth
settling before hardware is finalised. A 1080p or 4K native frame changes the answer
to the model question entirely.

Still run the sweep — the plateau point is useful for setting production `imgsz`
against compute cost. Just don't treat it as the resolution hypothesis being settled.

## 3. run_sahi's imgsz parameter is dead

```python
def run_sahi(image_path: str, imgsz: int = IMAGE_SIZE):
```

`imgsz` is never used in the body. SAHI's effective resolution comes from
`slice_height` / `slice_width` (320), which is unchanged.

So the run still isn't apples-to-apples: five YOLO models at 1280 vs SAHI at 320px
tiles. That may be the right comparison — tiling is the whole point of SAHI — but the
signature implies a configuration that isn't happening.

Either drop the parameter, or use it to set slice size and sweep that instead
(slice 320 vs 480 vs 640 is the meaningful SAHI knob).

## 4. Timings measure cold start, not steady state

Each `run_model()` constructs a fresh `YOLO()` and times a single `predict()`, so
every number includes first-inference warmup (graph init, lazy allocation).

Production uses a warm singleton via `get_detector()` and pays that cost once at boot.
These numbers will overstate real per-frame cost, possibly by a lot.

Fix: one throwaway `predict()` on the same image before starting the timer. Better,
time 3 runs after warmup and report the median.

This matters for the free-tier deploy decision — cold-start numbers could rule out a
model that's actually fine warm.

## 5. Band "recall" is a count ratio

```python
recall = detected / truth if truth else float("nan")
```

This divides two counts. It can exceed 100%, and a false positive inside a band
inflates it. It never matches individual detections against ground truth.

That's the same gap round 1 raised about scalar counts, reintroduced per band: it
can't distinguish "found the right 12 people" from "found 12 wrong things".

Banding is still worth having — it does expose spatial bias, which is the main
question. Just rename the column to `count ratio` so nobody reads it as true recall,
and note in the docstring that values over 100% mean false positives, not perfect
detection.

`GROUND_TRUTH_COUNTS` is still all `None`, so the script exits early. Needs a hand
count on one image before it produces anything.

Minor: band boundaries assume near/far maps to bottom/top thirds of the frame. True
for a typical wall mount, wrong for an overhead or angled one. Fine for now, revisit
against the real camera.

## 6. Queue images still missing

Round 1 item 8, still outstanding. This is the only item blocking me.

`b.jpg` has no serving line in it, so I can't trace the queue polygon or verify queue
count tracks line length. Density doesn't matter for this, queue visibility does. 2-3
frames with a visible counter line, or a pointer to where to find them.

Detection doesn't care about queues — that's entirely the zone layer. But I can't
build or validate that half against frames with no queue.

## Asks

- [ ] Run the corrected comparison and commit the output (counts + timings + annotated images)
- [ ] Revert `DEFAULT_CONFIDENCE_THRESHOLD` to 0.2, decide from sweep data
- [ ] Reword the "tuned against real mess-hall test photo" comment
- [ ] Drop or actually use `run_sahi`'s `imgsz`; consider sweeping slice size
- [ ] Warm up before timing, report median of 3
- [ ] Rename band `recall` to `count ratio`, document >100% case
- [ ] Fill `GROUND_TRUTH_COUNTS` for one image
- [ ] 2-3 images with visible queue
- [ ] Raise capture resolution with whoever is speccing the camera

Zone/metrics work continues against synthetic detections. Contract is stable and
nothing here blocks it except the queue images.
