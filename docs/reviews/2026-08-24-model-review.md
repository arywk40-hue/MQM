# Review notes: Human_detection/

Reviewed `model.py`, `model_rtdetr_sahi_compare.py`, and the five `annotated_*` outputs.

`model.py` is fine. Output contract is clean and I can build zones/metrics against it
now. Issues below are in the comparison script plus two unspecified contract details.

## 1. Comparison ran at the wrong resolution

`run_model()` doesn't pass `imgsz`, so all five models ran at the ultralytics default
of 640. `model.py` uses `DEFAULT_IMAGE_SIZE = 1280`. Conf also differs: 0.25 in the
comparison vs 0.2 in `model.py`.

The results don't describe the config we deploy.

Counts on `b.jpg` (approx, from the annotated images):

| Model | Count |
|---|---|
| YOLO11x | ~6 |
| YOLOv8n | ~10 |
| YOLOv8s | ~15 |
| RT-DETR-L | ~25 |
| SAHI + YOLOv8n | ~30 |

Ground truth is ~80-100. All five miss most of the room.

SAHI placed first running YOLOv8n on 320px tiles, beating YOLO11x by ~5x. Tiling
only increases effective pixels per person, so resolution looks like the binding
constraint rather than model capacity. A person at the back of the hall is roughly
15x30px in the source and doesn't survive the resize to 640. That's also consistent
with 11x scoring lowest, since it's better calibrated and won't guess on
unresolvable blobs.

Changes needed:

1. Add `imgsz` param to `run_model()`, pass through to `.predict()`.
2. `CONF_THRESHOLD = 0.2` to match `model.py`.
3. Re-run all models at 1280.
4. Sweep imgsz 640/960/1280/1920 on whichever model wins, print count per setting.
5. Test `augment=True`. Already supported in `model.py`, unused in the comparison.
   TTA is affordable at a ~10s cadence.
6. Print inference wall-time per model. We have no cost data and we're deploying to
   a free tier.

Expectation: YOLOv8s at 1280 lands near RT-DETR at 640 and the architecture spread
narrows. If it does, v8s is the pick. Hold off changing `DEFAULT_MODEL_WEIGHTS`
until the corrected run backs it.

## 2. Raw counts aren't a usable metric

A scalar count can't separate 30 correct detections from 20 correct plus 10 false
positives, and doesn't show which people were missed.

The misses aren't uniform, they're concentrated in the back of the room. That's
systematic bias and the count hides it.

Hand-label person counts for 3 horizontal bands (near/mid/far) on one image and
report per-band recall. "Uniformly mediocre" and "blind past 20 feet" have very
different downstream consequences for the zone layer.

## 3. Reference point: centroid vs bottom-center

`model.py` docstring says I derive centroid `((x1+x2)/2, (y1+y2)/2)` for zone tests.
Flagging that I'll likely change this rather than diverging silently.

Centroid sits at chest height. Under perspective it projects forward of the person's
actual floor position, which can put a seated person's reference point on the table
ahead of them or across a zone boundary. `((x1+x2)/2, y2)` is the floor contact point
and matches the question zones actually ask.

Tradeoff is that seated people have occluded legs, so `y2` is bottom-of-torso and
less stable.

This is zone logic, so my call. I'll make it a parameter defaulting to bottom-center.
No code change needed on your side, just loosen that docstring line so it doesn't
read as fixed. bbox format itself is unchanged.

## 4. Undocumented contract details

Add to the module docstring:

- Returns `[]` on no detections, never `None`. Empty hall is a normal case and
  `None` breaks the ingestion endpoint on every poll. Current code already returns
  `[]`; want it stated so a refactor doesn't change it quietly.
- Confirm boxes at `imgsz=1280` are in original-image pixels, not model-space.
  `box.xyxy` rescales by default so this is probably fine, but `zones.json` polygons
  have to be in the same space. If it's ever wrong, every point-in-polygon returns
  false and all metrics read zero with no error.

## 5. Test images

`b.jpg` works as a detector stress test. Keep it, add more at similar difficulty.

It has no queue in it, though, so I can't use it for zone work at all. Need 2-3
mess images with the serving line visible so I can trace the queue polygon and check
that queue count tracks line length. Density doesn't matter for that, visibility of
the queue does.

Detection doesn't care about queues, that's entirely my layer. But I can't build or
validate the queue zone from frames with no queue.

Note both sets are throwaway. Once we have a frame from the actual mounted camera it
replaces all of this for both of us, since your tuning and my polygons are both
geometry-dependent. Neither of us should over-fit to `b.jpg`. Getting that first real
frame is the biggest unblock available right now.

## Checklist

- [ ] `imgsz` param on `run_model()`, `CONF_THRESHOLD = 0.2`, re-run all at 1280
- [ ] imgsz sweep 640/960/1280/1920 on best model
- [ ] Test `augment=True`
- [ ] Report per-model inference time
- [ ] Per-band recall on one hand-labeled image
- [ ] Loosen centroid line in `model.py` docstring
- [ ] Document `[]`-on-empty and coordinate space
- [ ] Add 2-3 images with visible queue
- [ ] Don't change `DEFAULT_MODEL_WEIGHTS` until the corrected comparison justifies it

None of this blocks me. Contract is stable, I'm building zones/metrics against
synthetic detections meanwhile.
