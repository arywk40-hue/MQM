# Queue and seated classifier integration

## Promoted artifacts

- `models/queue_classifier.pt` — SHA-256
  `b1c74c30181139efbac1ab16bc6ff9275d5150737a2c9fa36e485a39d134188d`
- `models/seated_classifier.pt` — SHA-256
  `162502a05e8e9190d4a932b2b48a039dc486c2296592197b9f0001c0e0564609`

Both files were statically inspected with PyTorch's safe checkpoint loading.
They contain tensor-only MobileNetV2 state dictionaries with two output
classes, and load strictly into `torchvision.models.mobilenet_v2`.

## Label mapping

The source training folders establish alphabetical `ImageFolder` indices:

| Checkpoint | Class 0 | Class 1 | Production interpretation |
| --- | --- | --- | --- |
| Queue classifier | `not_queue` | `queue` | Class 1 is queued |
| Seated classifier | `seated` | `standing` | Class 0 is seated |

## Production behavior

YOLO remains the sole source of person detections and headcount. Each detected
person crop is resized to 224x224 RGB and ImageNet-normalized before both
classifiers run as a batch on CPU. The queue result can affect only a person
whose floor point is already in a configured queue zone; the seated result can
affect only a person in a configured seating zone. This prevents a classifier
from inventing a queue or seat outside the physical camera layout.

`ATTRIBUTE_CLASSIFIERS_ENABLED=true` activates the reviewed checkpoints. If a
checkpoint is missing or invalid, ingestion fails explicitly instead of
silently reverting to another model.

## Excluded source material

The `content/` handover contains raw crops, source photographs, and training
datasets (about 54 MB). It is ignored rather than committed: none are required
by the runtime, and they need separate provenance/privacy review before any
publication. Only the two reviewed production checkpoints were promoted.

## Verification

- Safe checkpoint inspection found no unsafe globals.
- Both checkpoints loaded strictly into MobileNetV2 and classified a test crop.
- Focused classifier/pipeline/metrics suite: 37 passed.
- Full suite: 63 passed, 1 opt-in test skipped in ordinary runs.
- Opt-in live stack with Redis, InfluxDB, FastAPI, YOLO, classifiers, and
  Streamlit: passed against an isolated local environment.
