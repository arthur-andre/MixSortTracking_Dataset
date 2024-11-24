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

# Run your Python script
python Json_dataset_creation_adj.py --match_name PAC_ATL

python Json_dataset_creation_adj.py --match_name POR_SAC

