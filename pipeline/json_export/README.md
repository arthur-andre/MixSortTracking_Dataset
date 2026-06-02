# JSON export

Turn MixSort outputs + per-frame 3D pose into **`players_data.json`**.

## Scripts

| File | Kalman on 3D pose |
|------|-------------------|
| `Json_dataset_creation.py` | No |
| `Json_dataset_creation_no_kalman.py` | No (ablation) |
| `Json_dataset_creation_adj.py` | Yes (`pykalman`) |

```shell
python pipeline/json_export/Json_dataset_creation_adj.py --match_name CHI_NYK
```

## Inputs

- `YOLOX_outputs/<match>/<action>/track_results/clip.txt`
- `nba_videos/<match>/<action>.mp4` (frame rate)
- `datasets/Input_Model/<match>/<action>/frame_<n>/npy/j3d.npy`

## Output

- `datasets/<match>/<action>/players_data.json`

Tracking must use `-expn <match>` so outputs live under `YOLOX_outputs/<match>/`.
