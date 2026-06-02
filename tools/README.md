# Tools

## This project (use `pipeline/` instead)

| Legacy wrapper | Preferred |
|----------------|-----------|
| `tools/preprocess_from_raw.py` | `pipeline/preprocess/from_raw.py` |
| `tools/preprocess_from_raw_model.py` | `pipeline/preprocess/from_raw_batch.py` |
| `tools/track_mixsort_local.py` | `pipeline/tracking/run_local.py` |
| `tools/track_mixsort.py` | `pipeline/tracking/run_batch.py` |

## Bundled from upstream MixSort

Training, ONNX export, MOT→COCO converters, interpolation, demo, etc. See [docs/MIXSORT_UPSTREAM.md](../docs/MIXSORT_UPSTREAM.md).
