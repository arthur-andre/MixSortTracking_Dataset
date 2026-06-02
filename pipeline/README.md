# Pipeline

All **new** code for MixSortTracking Dataset lives here. Upstream MixSort remains in `yolox/`, `MixViT/`, and `tools/` (train, deploy, etc.).

## Modules

| Directory | Entry point | Purpose |
|-----------|-------------|---------|
| [preprocess/](preprocess/) | `from_raw.py` | One MP4 → MOT layout + `test.json` |
| [preprocess/](preprocess/) | `from_raw_batch.py` | Batch preprocess a match folder |
| [tracking/](tracking/) | `run_local.py` | Track one clip (`-ofn`) |
| [tracking/](tracking/) | `run_batch.py` | Track all actions in a match (`--nba_match`) |
| [json_export/](json_export/) | `Json_dataset_creation*.py` | `clip.txt` + pose → `players_data.json` |
| [slurm/](slurm/) | `*.sh` | Example cluster jobs |

## Paths

Edit [paths.py](paths.py) or set `MIXSORT_*` environment variables (see root README).

## Import paths from tracking scripts

[\_repo.py](_repo.py) adds the repository root and `MixViT/` to `sys.path` — call `setup_import_paths()` before importing `yolox`.
