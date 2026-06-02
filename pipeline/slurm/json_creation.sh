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

python pipeline/json_export/Json_dataset_creation_adj.py --match_name PAC_ATL
python pipeline/json_export/Json_dataset_creation_adj.py --match_name POR_SAC
