# Changes from upstream MixSort

This repository is **not** a thin config fork. It is a **new project** that vendors [MixSort](https://github.com/MCG-NJU/MixSort) as the tracking engine and adds a full **video → tracks → ML dataset** stack for basketball (and similar) footage.

Base commit from upstream: `a078f5b` (MCG-NJU/MixSort).

---

## What this repo adds (does not exist in raw MixSort)

| Addition | Location | Role |
|----------|----------|------|
| Video → MOT/COCO preprocessing | `pipeline/preprocess/` | Extract frames, `test.json`, `seqinfo.ini` |
| Local + batch tracking drivers | `pipeline/tracking/` | `-ofn`, `--nba_match`, `sorted_actions.json` loops |
| JSON dataset export (3D pose, angles, motion) | `pipeline/json_export/` | `players_data.json` per clip |
| **3D pose Kalman smoothing** | `Json_dataset_creation_adj.py` | `pykalman` on joint trajectories (segment-aware) |
| Central path config | `pipeline/paths.py` | `MIXSORT_*` env vars, no cluster hardcoding |
| SLURM examples | `pipeline/slurm/` | Preprocess, track, export, full eval |
| Data layout docs | `datasets/`, `nba_videos/`, `output/` | Gitignored inputs/outputs with READMEs |
| Inference-first eval | `yolox/evaluators/mot_evaluator.py` | Skip MOT metrics when building datasets |
| Debug visualization | `yolox/utils/visualize.py` | Save `output/frame*.jpg` for demo GIF/MP4 |

---

## What we changed in upstream files

| File | Change |
|------|--------|
| `exps/example/mot/yolox_x_sportsmot.py` | Eval on `datasets/<clip>` via `MIXSORT_DATASETS`; `test.json` |
| `yolox/mixsort_tracker/matching.py` | `np.float64` / `np.int64` (NumPy 1.24+) |
| `yolox/mixsort_tracker/mixsort_tracker.py` | Same dtype fixes |
| `MixViT/lib/models/mixformer_vit/pos_util.py` | `np.float64` |
| `requirements.txt` | `pykalman`, `pandas`; relaxed torch pins |
| `.gitignore` | Ignore `datasets/`, `nba_videos/`, `output/`, `YOLOX_outputs/` |

---

## What is unchanged (still upstream MixSort)

- MixSort association logic (`yolox/mixsort_tracker/`) including **box Kalman** for bounding boxes  
- YOLOX detector, MixFormer (`MixViT/`), training scripts in `tools/train.py`  
- OC-SORT variant, TensorRT/ncnn deploy, TrackEval submodule  
- SportsMOT training/evaluation workflow → [MIXSORT_UPSTREAM.md](MIXSORT_UPSTREAM.md)

---

## Kalman: two different mechanisms

1. **Box Kalman (upstream)** — smooths 2D bounding boxes during tracking (`yolox/mixsort_tracker/kalman_filter.py`). Present in both repos.  
2. **Pose Kalman (this repo only)** — smooths **3D skeleton** sequences in `Json_dataset_creation_adj.py` after an external pose estimator writes `j3d.npy` files. Uses `pykalman.KalmanFilter.smooth()` on consecutive frame segments (gap &gt; 5 frames → new segment).

---

## Repository layout vs MixSort

```
MixSort (upstream)                 MixSortTracking_Dataset (this repo)
─────────────────                ─────────────────────────────────────
tools/track_mixsort.py      →    pipeline/tracking/run_batch.py (+ wrapper)
                                 pipeline/preprocess/
                                 pipeline/json_export/
                                 pipeline/slurm/
                                 assets/demo_compressed.gif
README = SportsMOT only       →  README = this pipeline first
```
