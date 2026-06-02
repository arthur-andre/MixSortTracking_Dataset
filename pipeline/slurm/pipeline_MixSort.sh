#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-12:00
#SBATCH -p gpu_test
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err

module load cuda/11.3.1-fasrc01
source activate nba_bis

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

python pipeline/preprocess/from_raw.py \
  nba_videos/2s/fultz_2s.mp4 \
  datasets/fultz_2s \
  injury_fultz

python pipeline/tracking/run_local.py \
  -expn fultz_2s \
  -f exps/example/mot/yolox_x_sportsmot.py \
  -c pretrained/yolox_x_sports_train.pth.tar \
  -b 1 -d 1 \
  -ofn fultz_2s/test/injury_fultz \
  --config track
