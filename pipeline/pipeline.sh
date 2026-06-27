#!/bin/bash

# Define the path to the YAML file
YAML_FILE_POSE="img_to_mesh/src/experiments/pose/pose_demo.yaml"
YAML_FILE_MESH="img_to_mesh/src/experiments/mesh/mesh_demo.yaml"

# Define new values
NEW_STARTING_FRAME=1
NEW_ENDING_FRAME=157
NEW_OUTPUT_NAME="lbj_clip"

# Use sed to replace the values
sed -i "s/^  starting_frame: .*/  starting_frame: $NEW_STARTING_FRAME/" $YAML_FILE_POSE
sed -i "s/^  ending_frame: .*/  ending_frame: $NEW_ENDING_FRAME/" $YAML_FILE_POSE
sed -i "s/^  output_name: .*/  output_name: $NEW_OUTPUT_NAME/" $YAML_FILE_POSE

echo "Options in $YAML_FILE_POSE have been updated."

sed -i "s/^  starting_frame: .*/  starting_frame: $NEW_STARTING_FRAME/" $YAML_FILE_MESH
sed -i "s/^  ending_frame: .*/  ending_frame: $NEW_ENDING_FRAME/" $YAML_FILE_MESH
sed -i "s/^  output_name: .*/  output_name: $NEW_OUTPUT_NAME/" $YAML_FILE_MESH

echo "Options in $YAML_FILE_MESH have been updated."

cd img_to_mesh/src

# Run the pose estimation
bash experiments/pose/pose_run.sh

# Run the mesh generation
bash experiments/mesh/mesh_run.sh


