#!/bin/bash
#SBATCH -c 2                        # Number of CPU cores
#SBATCH -t 0-12:00                  # Runtime in D-HH:MM (12 hours max)
#SBATCH -p gpu_test                 # Partition to submit to
#SBATCH --mem=12000                 # Memory pool for all cores (12GB)
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH -o myoutput_%j.out          # File to which STDOUT will be written (%j inserts jobid)
#SBATCH -e myerrors_%j.err          # File to which STDERR will be written (%j inserts jobid)
#SBATCH --mail-type=BEGIN,END,FAIL  # Types of email notifications (optional)
#SBATCH --mail-user=aandre@seas.harvard.edu  # Your email for notifications (optional)

# Load the required modules
module load cuda/11.3.1-fasrc01 || { echo "Failed to load CUDA module"; exit 1; }

# Activate the Conda environment
source activate nba_bis 

echo "begin of the evaluation"

# Define the path to the YAML files
YAML_FILE_POSE="../NBA-Players/img_to_mesh/src/experiments/pose/pose_demo.yaml"

# List of video files
videos=("nba_videos/CHI_NYK/7.mp4" "nba_videos/CHI_NYK/9.mp4" "nba_videos/CHI_NYK/12.mp4" "nba_videos/CHI_NYK/14.mp4" "nba_videos/CHI_NYK/15.mp4" "nba_videos/CHI_NYK/17.mp4" "nba_videos/CHI_NYK/19.mp4" "nba_videos/CHI_NYK/25.mp4" "nba_videos/CHI_NYK/162.mp4" "nba_videos/CHI_NYK/152.mp4")  # Add your video file paths here

# Loop over each video file
for video in "${videos[@]}"; do
    # Extract the video name without extension for output folder names
    video_name=$(basename "$video" .mp4)
    
    NEW_OUTPUT_NAME="evaluation"
    NEW_SEQUENCE_NAME="action"
    
    # Use sed to replace the values in pose_demo.yaml
    sed -i "s/^  output_name: .*/  output_name: $NEW_OUTPUT_NAME/" $YAML_FILE_POSE || { echo "Failed to update YAML file"; exit 1; }
    echo "Options in $YAML_FILE_POSE have been updated for $video_name."
    
    # Run preprocessing
    python tools/preprocess_from_raw.py "$video" "datasets/$NEW_OUTPUT_NAME" "$NEW_SEQUENCE_NAME" || { echo "Preprocessing failed for $video"; exit 1; }
    
    # Run tracking with MixSort
    python tools/track_mixsort.py -expn on_teste -f exps/example/mot/yolox_x_sportsmot.py -c pretrained/yolox_x_sports_train.pth.tar -b 1 -d 1 -ofn $NEW_OUTPUT_NAME --config track || { echo "MixSort tracking failed for $video"; exit 1; }
    
    # Navigate to NBA-Players and run preprocessing
    cd ../NBA-Players || { echo "Failed to change directory to NBA-Players"; exit 1; }
    python preprocess/MixSortPreprocess.py --out_fold_name $NEW_OUTPUT_NAME --sequence_name $NEW_SEQUENCE_NAME || { echo "MixSortPreprocess failed for $video"; exit 1; }
    
    # Navigate back to img_to_mesh/src and run pose estimation
    cd img_to_mesh/src || { echo "Failed to change directory to img_to_mesh/src"; exit 1; }
    bash experiments/pose/pose_run.sh || { echo "Pose estimation script failed for $video"; exit 1; }
    
    # Navigate back and run Kalman smoothing
    cd ../../ || { echo "Failed to change directory"; exit 1; }
    python get_input_kalman.py --name_sequence $NEW_SEQUENCE_NAME --clip_name $NEW_OUTPUT_NAME || { echo "get_input_kalman failed for $video"; exit 1; }
    
    # Pass the video name to the final Python script
    python update_poses_kalman.py --clip_name $NEW_OUTPUT_NAME --eval True --video_name $video_name || { echo "update_poses_kalman failed for $video"; exit 1; }
    
    cd ../MixSortTracking

    echo "Processing completed for $video"
done

echo "All videos have been processed."
