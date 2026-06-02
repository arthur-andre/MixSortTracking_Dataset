# Upstream MixSort (SportsMOT)

This document summarizes the **original** [MixSort](https://github.com/MCG-NJU/MixSort) project bundled in this repo. For the **NBA / custom JSON dataset pipeline**, use the root [README.md](../README.md).

## Overview

MixSort improves appearance-based association for tracking-by-detection systems (ByteTrack, OC-SORT) using a MixFormer appearance model. It is the baseline tracker for [SportsMOT](https://github.com/MCG-NJU/SportsMOT).

See the upstream repo for architecture diagrams and SportsMOT result GIFs.

## Installation (upstream)

```shell
git clone --recursive https://github.com/MCG-NJU/MixSort
cd MixSort
conda create -n MixSort python=3.8
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
python setup.py develop
pip install cython pycocotools cython_bbox
pip install -r MixViT/requirements.txt
```

## Standard data layout (SportsMOT, MOT17, …)

```
datasets/
├── SportsMOT/{train,val,test}
├── MOT17/{train,test}
└── crowdhuman/...
```

Convert to COCO format:

```shell
python tools/convert_sportsmot_to_coco.py
python tools/convert_mot17_to_coco.py
# ... other convert_* scripts
```

## Tracking (SportsMOT)

```shell
python tools/track_mixsort.py \
  -expn sports_demo \
  -f exps/example/mot/yolox_x_sportsmot.py \
  -c pretrained/yolox_x_sports_train.pth.tar \
  -b 1 -d 1 \
  --config track
```

Set `self.val_ann` in the exp file to `val.json` or `train.json` as needed. For OC-SORT variant, use `tools/track_mixsort_oc.py`.

## Training

See the [MixSort README](https://github.com/MCG-NJU/MixSort#training) for YOLOX and MixFormer training on CrowdHuman, MOT datasets, and custom data.

## Evaluation (TrackEval)

```shell
git submodule update --init --recursive
ln -s datasets/SportsMOT/val TrackEval/data/gt/mot_challenge/sports-val
python TrackEval/scripts/run_mot_challenge.py --BENCHMARK sports --SPLIT_TO_EVAL val --TRACKERS_TO_EVAL <exp_name>
```

## Model zoo

Pretrained weights: [Google Drive](https://drive.google.com/drive/folders/1pQs1gFC_jG0TlGIUMgf3E0I3OztCvgxI?usp=sharing) — place under `pretrained/`.

## Citation

```bibtex
@article{cui2023sportsmot,
  title={SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes},
  author={Cui, Yutao and others},
  journal={arXiv preprint arXiv:2304.05170},
  year={2023}
}
```
