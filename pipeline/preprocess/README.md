# Preprocess

Convert a raw video into MixSort’s expected MOT + COCO layout.

```shell
python pipeline/preprocess/from_raw.py <video.mp4> <datasets_root> <sequence_name>
```

Example:

```shell
python pipeline/preprocess/from_raw.py nba_videos/CHI_NYK/7.mp4 datasets/CHI_NYK action
```

Creates:

- `datasets/CHI_NYK/test/action/img1/000001.jpg …`
- `datasets/CHI_NYK/annotations/test.json`
- `seqinfo.ini` with fps and frame count
