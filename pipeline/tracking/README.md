# Tracking

MixSort inference drivers (modified from upstream `tools/track_mixsort.py`).

## Single clip

```shell
python pipeline/tracking/run_local.py \
  -expn <match_name> \
  -ofn <match_name>/test/<sequence> \
  -f exps/example/mot/yolox_x_sportsmot.py \
  -c pretrained/yolox_x_sports_train.pth.tar \
  -b 1 -d 1 --config track
```

## Full match (many actions)

Requires `datasets/<match>/sorted_actions.json` and preprocessed folders per action.

```shell
python pipeline/tracking/run_batch.py \
  --nba_videos True --nba_match CHI_NYK \
  -expn CHI_NYK \
  -f exps/example/mot/yolox_x_sportsmot.py \
  -c pretrained/yolox_x_sports_train.pth.tar \
  -b 1 -d 1 --config track
```

Outputs: `YOLOX_outputs/<match>/<action>/track_results/clip.txt`
