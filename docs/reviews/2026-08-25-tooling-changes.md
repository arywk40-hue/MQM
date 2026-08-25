# Change note: comparison tooling fixes

The round-2 script issues have been applied directly rather than raised as a further
review round. Modified in `Updated/`:

- `model_rtdetr_sahi_compare.py`
- `eval_band_recall.py`

`model.py` is unmodified. Pull these changes before making further edits.

## Applied changes

### Inference timing now measures steady state

`run_model()` timed a single `predict()` against a freshly constructed `YOLO()`
instance, so every measurement included first-inference warmup. Production serves from
a warm singleton via `get_detector()` and incurs that cost once at startup, meaning
cold-start figures materially overstate per-frame cost — potentially enough to exclude
a model that performs acceptably when warm.

Revised to discard one warmup prediction on the same code path (`augment` is forwarded,
as TTA follows a separate path), then report the median of three timed runs.

### `run_sahi()` resolution parameter corrected

The function accepted an `imgsz` argument that was never referenced. SAHI's effective
resolution is determined by `slice_height` / `slice_width`, so the signature advertised
a configuration that was not being applied.

Replaced with `slice_size`, and the entry point now sweeps 320/480/640. Output
filenames include the slice size to prevent runs overwriting one another.

### Band metric relabelled

`detected / truth` is a ratio of two counts. It can exceed 100%, and a false positive
within a band inflates it. Because detections are never matched to individual
ground-truth subjects, it cannot distinguish correct detections from an equal number of
incorrect ones — the precise limitation the script was introduced to address.

The column is now labelled "count ratio", with a printed note clarifying that values
above 100% indicate false positives rather than complete detection. The metric remains
appropriate for its actual purpose, which is identifying spatial bias.

### Deliberately unchanged

`DEFAULT_CONFIDENCE_THRESHOLD` remains at 0.25. Round 2 recommended reverting to 0.2;
the sweep should settle this empirically rather than by argument.

## Outstanding

**1. Execute the comparison.** The scripts are corrected but have produced no data — no
`annotated_*_1280.jpg`, no timings, no sweep output. The model selection question is
unresolved. Please commit console output alongside the generated images.

**2. Populate `GROUND_TRUTH_COUNTS` in `eval_band_recall.py`.** Requires a manual count
of subjects in the near/mid/far thirds of one image. The script exits early until this
is supplied.

This result has direct downstream consequences. If detection recall collapses beyond
approximately 20 feet and the serving counter is positioned toward the rear of the
frame, queue count becomes systematically understated while seat occupancy remains
accurate — one of four reported metrics silently incorrect, with no error surfaced. The
per-band figures are what expose this.

**3. Revise the confidence threshold comment.** It states the value was "tuned against
real mess-hall test photo". No production footage exists; `b.jpg` is a sourced image.
The comment will mislead whoever re-tunes this after camera installation.

**4. Test imagery with a visible queue.** Outstanding from round 1. `b.jpg` contains no
serving line, so the queue polygon cannot be validated against it. Two or three frames
showing a counter queue, or a pointer to a source. Crowd density is not the requirement;
queue visibility is.

## Items for wider team decision

**Capture resolution is unspecified.** `b.jpg` is 736x490. Running it at `imgsz=1280` or
1920 upscales beyond native resolution, so the sweep cannot cleanly establish the
resolution hypothesis. Some improvement is expected, since YOLO detects at fixed anchor
scales and upscaling a 15x30px subject to 26x52 can cross the detectable threshold —
but that should be read as the network preferring larger input, not as additional detail
contributing.

The binding constraint is the resolution the mounted camera captures at, which has not
been specified. A 1080p or 4K native frame materially changes the model selection
answer. This warrants resolution before hardware is finalised.

**Reference-point selection is resolved.** The zone layer defaults to bottom-center
`((x1+x2)/2, y2)`, configurable to centroid, implemented as a parameter in
`inference/zones.py`. Test coverage demonstrates the two modes assigning the same
subject to different zones where feet fall inside a zone but torso centre sits above its
upper edge. The current `model.py` docstring already reflects this correctly; no change
required.

## Interface status

The detection interface is stable and unblocked:

```python
[{"bbox": [x1, y1, x2, y2], "confidence": 0.87}, ...]
```

Corner coordinates in original-image pixel space, empty list when no detections.

The downstream zone and metrics layer is implemented and tested against this interface —
`inference/zones.py`, `inference/metrics.py`, `config/zones.json`, 28 tests passing
against synthetic detections with no ultralytics dependency. Behaviour is identical
regardless of which model is selected, so model selection is not a blocker and has not
been one.

The remaining dependencies are test imagery containing a queue (item 4) and a frame from
the production camera.
