<h1 align="center">MixSortTracking Dataset</h1>

<p align="center">
  <strong>Vidéo sport → suivi multi-joueurs → datasets JSON pour le ML</strong><br>
  Extension de <a href="https://github.com/MCG-NJU/MixSort">MixSort</a> (YOLOX + MixFormer) pour clips NBA et footage personnalisée.
</p>

<p align="center">
  <img src="assets/demo_compressed.gif" alt="Suivi MixSort sur un clip NBA (CHI vs NYK) — boîtes et ID joueurs" width="800"/>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#utilisation">Utilisation</a> ·
  <a href="#structure">Structure</a> ·
  <a href="docs/CHANGES_FROM_MIXSORT.md">Différences vs MixSort</a> ·
  <a href="docs/MIXSORT_UPSTREAM.md">MixSort d'origine</a>
</p>

---

## En bref

Ce dépôt transforme des **vidéos brutes** (`nba_videos/`) en **fichiers `players_data.json`** : une entrée par joueur suivi, avec pose 3D, angles, vitesses et option de lissage Kalman sur les articulations.

| Étape | Script | Sortie |
|-------|--------|--------|
| 1. Préparer les images | `pipeline/preprocess/from_raw.py` | `datasets/…/img1/` + `test.json` |
| 2. Suivre les joueurs | `pipeline/tracking/run_local.py` | `YOLOX_outputs/…/clip.txt` |
| 3. Exporter le dataset | `pipeline/json_export/Json_dataset_creation_adj.py` | `players_data.json` |

Le code ajouté par ce projet vit dans **`pipeline/`**. Le reste (`yolox/`, `MixViT/`, `tools/`) est le moteur MixSort d'origine.

---

## Installation

```bash
git clone --recursive https://github.com/arthur-andre/MixSortTracking_Dataset.git
cd MixSortTracking_Dataset

conda create -n mixsort-dataset python=3.8 && conda activate mixsort-dataset
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

pip install -r requirements.txt && python setup.py develop
pip install cython pycocotools cython_bbox
pip install -r MixViT/requirements.txt
```

Télécharger les poids dans `pretrained/` : `yolox_x_sports_train.pth.tar` + checkpoint MixFormer ([zoo MixSort](https://github.com/MCG-NJU/MixSort#model-zoo)).

Chemins personnalisés : voir [.env.example](.env.example) (`MIXSORT_DATASETS`, `MIXSORT_NBA_VIDEOS`, …).

---

## Utilisation

### 1 — Vidéo → images MOT

```bash
python pipeline/preprocess/from_raw.py \
  nba_videos/CHI_NYK/7.mp4 \
  datasets/CHI_NYK \
  action
```

### 2 — Suivi MixSort

Utiliser **`-expn`** = nom du match (ex. `CHI_NYK`) pour que l'export JSON retrouve les résultats.

```bash
python pipeline/tracking/run_local.py \
  -expn CHI_NYK \
  -f exps/example/mot/yolox_x_sportsmot.py \
  -c pretrained/yolox_x_sports_train.pth.tar \
  -b 1 -d 1 \
  -ofn CHI_NYK/test/action \
  --config track
```

Plusieurs clips d'un même match : `pipeline/tracking/run_batch.py` avec `--nba_videos True --nba_match CHI_NYK`.

### 3 — Export JSON

```bash
python pipeline/json_export/Json_dataset_creation_adj.py --match_name CHI_NYK
```

| Script | Lissage Kalman 3D |
|--------|-------------------|
| `Json_dataset_creation.py` | Non |
| `Json_dataset_creation_no_kalman.py` | Non |
| `Json_dataset_creation_adj.py` | **Oui** |

Nécessite les poses 3D par frame dans `datasets/Input_Model/<match>/<action>/frame_<n>/npy/`.

---

## Structure

```
MixSortTracking_Dataset/
├── assets/demo_compressed.gif   # Démo README (suivi + ID)
├── pipeline/                # ★ Code de ce projet
│   ├── preprocess/
│   ├── tracking/
│   ├── json_export/
│   └── slurm/
├── nba_videos/              # Entrée MP4 (gitignore)
├── datasets/                # Clips + JSON (gitignore)
├── YOLOX_outputs/           # Résultats de suivi (gitignore)
├── yolox/ · MixViT/         # MixSort (amont)
└── docs/
```

Détails : [pipeline/README.md](pipeline/README.md) · [datasets/README.md](datasets/README.md)

---

## Citation & licence

MixSort / SportsMOT — [article](https://arxiv.org/abs/2304.05170). Licence MIT : [LICENSE](LICENSE) · [NOTICE.md](NOTICE.md).
