#!/bin/bash
#SBATCH -c 2                        # Number of CPU cores
#SBATCH -t 0-12:00                  # Runtime in D-HH:MM (12 hours max)
#SBATCH -p gpu_test                 # Partition to submit to
#SBATCH --mem=12000                # Memory pool for all cores (240GB)
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH -o myoutput_%j.out          # File to which STDOUT will be written (%j inserts jobid)
#SBATCH -e myerrors_%j.err          # File to which STDERR will be written (%j inserts jobid)
#SBATCH --mail-type=BEGIN,END,FAIL  # Types of email notifications (optional)
#SBATCH --mail-user=aandre@seas.harvard.edu  # Your email for notifications (optional)

# Load the required modules
module load cuda/11.3.1-fasrc01

# Activate the Conda environment
source activate nba_bis

python tools/preprocess_from_raw.py nba_videos/2s/fultz_2s.mp4 datasets/fultz_2s injury_fultz

python tools/track_mixsort.py -expn on_teste -f exps/example/mot/yolox_x_sportsmot.py -c pretrained/yolox_x_sports_train.pth.tar -b 1 -d 1 -ofn fultz_2s --config track

