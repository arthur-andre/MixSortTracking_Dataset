#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-12:00
#SBATCH -p gpu_test
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
# Requires sibling repo: ../NBA-Players

module load cuda/11.3.1-fasrc01 || exit 1
source activate nba_bis

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NBA_PLAYERS_ROOT="$(cd "$REPO_ROOT/../NBA-Players" 2>/dev/null && pwd)"

cd "$REPO_ROOT" || exit 1

YAML_FILE_POSE="${NBA_PLAYERS_ROOT}/img_to_mesh/src/experiments/pose/pose_demo.yaml"
videos=("nba_videos/CHI_NYK/7.mp4" "nba_videos/CHI_NYK/9.mp4")

for video in "${videos[@]}"; do
  video_name=$(basename "$video" .mp4)
  NEW_OUTPUT_NAME="evaluation"
  NEW_SEQUENCE_NAME="action"

  sed -i "s/^  output_name: .*/  output_name: $NEW_OUTPUT_NAME/" "$YAML_FILE_POSE" || exit 1

  python pipeline/preprocess/from_raw.py \
    "$video" "datasets/$NEW_OUTPUT_NAME" "$NEW_SEQUENCE_NAME" || exit 1

  python pipeline/tracking/run_local.py \
    -expn on_teste \
    -f exps/example/mot/yolox_x_sportsmot.py \
    -c pretrained/yolox_x_sports_train.pth.tar \
    -b 1 -d 1 \
    -ofn "$NEW_OUTPUT_NAME/test/$NEW_SEQUENCE_NAME" \
    --config track || exit 1

  cd "$NBA_PLAYERS_ROOT" || exit 1
  python preprocess/MixSortPreprocess.py \
    --out_fold_name "$NEW_OUTPUT_NAME" \
    --sequence_name "$NEW_SEQUENCE_NAME" || exit 1

  cd img_to_mesh/src || exit 1
  bash experiments/pose/pose_run.sh || exit 1

  cd "$NBA_PLAYERS_ROOT" || exit 1
  python get_input_kalman.py \
    --name_sequence "$NEW_SEQUENCE_NAME" \
    --clip_name "$NEW_OUTPUT_NAME" || exit 1
  python update_poses_kalman.py \
    --clip_name "$NEW_OUTPUT_NAME" \
    --eval True \
    --video_name "$video_name" || exit 1

  cd "$REPO_ROOT" || exit 1
  echo "Processing completed for $video"
done

echo "All videos have been processed."
