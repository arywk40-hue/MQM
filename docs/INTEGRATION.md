# Integrating code from teammates

How a file handed over by another team member becomes part of the tracked
codebase. The goal is that nobody edits production source by dropping a folder
into the repository and hoping.

## Repository layout

```
MQM/
├── config/zones.json           camera zones, capacity, thresholds
├── inference/
│   ├── model.py                person detection            (owner: detection)
│   ├── zones.py                zone assignment             (owner: metrics)
│   └── metrics.py              reported figures            (owner: metrics)
├── api/                        ingestion + read endpoints  (owner: API)
├── dashboard/                  Streamlit UI                (owner: dashboard)
├── tests/                      runs without ultralytics installed
├── tools/
│   ├── draw_zones.py           render zones over a frame
│   └── benchmarks/             throwaway evaluation scripts, not deployed
├── data/
│   ├── samples/                committed test images
│   └── outputs/                generated artefacts (ignored)
├── models/                     YOLO cache plus reviewed production classifiers
├── docs/reviews/               review history
└── incoming/                   staging for handovers (ignored)
```

Three rules the layout enforces:

- **`inference/` is production.** Everything in it is tracked, imported by the
  API, and covered by tests.
- **`tools/benchmarks/` is disposable.** Evaluation scripts that inform
  decisions but never run in production. Keeping them out of `inference/`
  prevents a throwaway script from being mistaken for a deployed component.
- **Large and generated files are not committed.** Weights, annotated outputs,
  and staged handovers are all ignored. The repository stays clonable.

## Receiving a file

**1. Drop it in `incoming/`, unmodified.**

```bash
mkdir -p incoming/2026-08-25-model
cp ~/Downloads/model.py incoming/2026-08-25-model/
```

`incoming/` is gitignored. Nothing here affects the build, so an incoming file
cannot break anything before it has been read.

**2. Diff it against what is already in the tree.**

```bash
git diff --no-index inference/model.py incoming/2026-08-25-model/model.py
```

This is the step that matters. It answers "what actually changed" rather than
"what was I told changed" — those diverged twice already in this project. A
confidence threshold moved from 0.2 to 0.25 in a change described as
documentation-only.

Read the whole diff. Anything altering a constant, a default argument, or a
return shape affects downstream code even when the description says otherwise.

**3. Check the interface has not moved.**

`inference/model.py` must keep returning:

```python
[{"bbox": [x1, y1, x2, y2], "confidence": 0.87}, ...]
```

Corner coordinates in original-image pixel space, empty list when nothing is
detected. `inference/zones.py` and the API's pydantic models both depend on
this. If a change touches it, that is a conversation before it is a merge.

**4. Promote it.**

```bash
cp incoming/2026-08-25-model/model.py inference/model.py
```

**5. Run the tests.**

```bash
python tests/test_zones_metrics.py
```

The zone and metrics tests use synthetic detections and no ultralytics, so they
pass or fail on interface compatibility alone. A failure here means the
detection output shape changed.

**6. Smoke-test the detector itself** if the change touched detection:

```bash
python -m inference.model data/samples/mess_hall_dense.jpg
```

**7. Commit with the origin recorded.**

```bash
git add inference/model.py
git commit -m "inference: update detector from <name> handover

<what changed and why, including anything the handover note omitted>"
```

**8. Clear the staging directory** once promoted, so `incoming/` never becomes
a second copy of the codebase.

## Sending changes back

When editing someone else's file, keep the change and the explanation together.
Write the change note to `docs/reviews/YYYY-MM-DD-<topic>.md`, describing what
changed and why, and hand over both the file and the note.

The reviews already in `docs/reviews/` follow this pattern.

## Weights

Standard detector weights are not in version control. `models/yolov8n.pt` is
ignored and downloaded by Ultralytics if absent. The two reviewed, small
production attribute classifiers are the deliberate exception:
`models/queue_classifier.pt` and `models/seated_classifier.pt` are tracked.
Raw crops, source photos, and training datasets are not production assets and
remain ignored under `content/`.

Ultralytics downloads standard weights on first use, so a fresh clone works
without any manual step. `inference.model.resolve_weights()` checks `models/`
first and falls through to that download if the file is absent.

To pre-populate:

```bash
python -c "from ultralytics import YOLO; [YOLO(w) for w in ['yolov8n.pt','yolov8s.pt']]"
```

## Environment

```bash
python -m pip install -r requirements.txt
```

`inference/zones.py` and `inference/metrics.py` deliberately have no
third-party dependencies, so the metrics layer can be developed and tested
without a working detection environment.

## Legacy folders

`Human_detection/` is the original prototype directory, superseded by
`inference/` and ignored. It is retained temporarily so nothing is lost during
the transition and should be deleted once the team confirms nothing references
it.
