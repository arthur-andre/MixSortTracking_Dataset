import numpy as np
import json
from collections import defaultdict
import pandas as pd
import cv2
import os
import argparse
from math import pi

def center_pose(joints_3d, reference_idx):
    """
    Center the pose by subtracting the reference joint (e.g., pelvis) position from all joints.
    
    Args:
    joints_3d: (35, 3) array of joint positions.
    reference_idx: Index of the joint to center on (e.g., pelvis).
    
    Returns:
    centered_joints: (35, 3) array of centered joint positions.
    """
    reference_point = joints_3d[reference_idx]
    centered_joints = joints_3d - reference_point
    return centered_joints

def align_hips_to_x_axis(joints_3d, left_hip_idx, right_hip_idx):
    """
    Align the left and right hips to the x-axis by rotating the pose.
    
    Args:
    joints_3d: (35, 3) array of joint positions.
    left_hip_idx: Index of the left hip joint.
    right_hip_idx: Index of the right hip joint.
    
    Returns:
    aligned_joints: (35, 3) array of aligned joint positions.
    """
    # Compute the vector between the left and right hip
    hip_vector = joints_3d[right_hip_idx] - joints_3d[left_hip_idx]
    
    # Normalize the hip vector
    hip_vector = hip_vector / np.linalg.norm(hip_vector)

    
    # The target vector is the unit vector along the x-axis
    target_vector = np.array([1, 0, 0])
    
    # Compute the axis of rotation as the cross product of the hip_vector and target_vector
    axis_of_rotation = np.cross(hip_vector, target_vector)

    
    # If the vector is already aligned, no rotation is needed
    if np.linalg.norm(axis_of_rotation) < 1e-6:
        return joints_3d
    
    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)
    
    # Compute the angle between the hip vector and the target vector
    angle = np.arccos(np.dot(hip_vector, target_vector))
    
    # Construct the rotation matrix using the axis-angle formula
    R = rotation_matrix_from_axis_angle(axis_of_rotation, angle)
    
    # Apply the rotation matrix to all joints
    aligned_joints = np.dot(R, joints_3d.T).T
    
    return aligned_joints

def rotation_matrix_from_axis_angle(axis, angle):
    """
    Construct a rotation matrix from an axis and an angle using Rodrigues' rotation formula.
    
    Args:
    axis: The axis of rotation (3,).
    angle: The angle of rotation in radians.
    
    Returns:
    R: The rotation matrix (3, 3).
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1 - cos_angle
    
    x, y, z = axis
    R = np.array([
        [cos_angle + x*x*one_minus_cos, x*y*one_minus_cos - z*sin_angle, x*z*one_minus_cos + y*sin_angle],
        [y*x*one_minus_cos + z*sin_angle, cos_angle + y*y*one_minus_cos, y*z*one_minus_cos - x*sin_angle],
        [z*x*one_minus_cos - y*sin_angle, z*y*one_minus_cos + x*sin_angle, cos_angle + z*z*one_minus_cos]
    ])
    
    return R

def align_hips_by_z_rotation(joints_3d, left_hip_idx, right_hip_idx, pelvis_idx):
    """
    Center the pose at the pelvis and rotate the pose around the z-axis so that the hip vector
    (between the left and right hip) is aligned with the x-axis (removes y-component).
    
    Args:
    joints_3d: (N, 3) array of joint positions.
    left_hip_idx: Index of the left hip joint.
    right_hip_idx: Index of the right hip joint.
    pelvis_idx: Index of the pelvis joint.
    
    Returns:
    aligned_joints: (N, 3) array of aligned and centered joint positions.
    """
    # Step 1: Center the pose at the pelvis
    pelvis_pos = joints_3d[pelvis_idx]
    joints_3d_centered = joints_3d - pelvis_pos  # Move pelvis to the origin

    # Step 2: Compute the hip vector (right hip - left hip)
    hip_vector = joints_3d_centered[right_hip_idx] - joints_3d_centered[left_hip_idx]

    # Step 3: Calculate the angle to rotate around the z-axis to remove the y-component of the hip vector
    angle = np.arctan2(hip_vector[1], hip_vector[0])  # atan2(y, x) gives the angle to the x-axis

    # Step 4: Create the z-axis rotation matrix to rotate by -angle (to align with the x-axis)
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)
    R_z = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])

    # Step 5: Apply the rotation to all joints
    aligned_joints = np.dot(R_z, joints_3d_centered.T).T  # Rotate all joints

    return aligned_joints

def normalize_pose_orientation(joints_3d, left_hip_idx, right_hip_idx, pelvis_idx):
    """
    Normalize the pose orientation by centering the pelvis at the origin and aligning the hips to the x-axis.
    
    Args:
    joints_3d: (35, 3) array of joint positions.
    left_hip_idx: Index of the left hip joint.
    right_hip_idx: Index of the right hip joint.
    pelvis_idx: Index of the pelvis joint.
    
    Returns:
    normalized_joints: (35, 3) array of normalized joint positions.
    """
    # Step 1: Center the pose around the pelvis
    centered_joints = center_pose(joints_3d, pelvis_idx)
    
    # Step 2: Align the left and right hips to the x-axis
    aligned_joints = align_hips_to_x_axis(centered_joints, left_hip_idx, right_hip_idx)
    
    return aligned_joints

def normalize_poses_orientationless(players_poses, left_hip_idx, right_hip_idx, pelvis_idx):
    """
    Normalize a set of player poses to be orientationless by centering the pelvis at the origin
    and aligning the hips to the x-axis.
    
    Args:
    player_pose: (35, 3) array of poses.
    left_hip_idx, right_hip_idx, pelvis_idx: Indices for relevant joints.
    
    Returns:
    normalized_pose: (35, 3) array of normalized poses.
    """
    aligned_joints = align_hips_by_z_rotation(players_poses, left_hip_idx=4, right_hip_idx=1, pelvis_idx=0)

    return np.array(aligned_joints)

def calculate_angle_3d(A, B, C):
    """
    Calculate the angle at point B given three points A, B, C in 3D space.
    """
    # Convert points to numpy arrays
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # Vectors BA and BC
    BA = A - B
    BC = C - B
    
    # Check if either vector is a zero vector
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)
    
    if norm_BA == 0 or norm_BC == 0:
        return np.nan  # Return NaN if we cannot compute the angle due to zero vectors
    
    # Calculate the cosine of the angle using dot product and magnitudes
    cosine_angle = np.dot(BA, BC) / (norm_BA * norm_BC)
    
    # Clamp cosine_angle to avoid rounding errors outside the range [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate the angle in radians, then convert to degrees
    angle_rad = np.arccos(cosine_angle)
    
    return angle_rad


def calculate_angle_2d(A, B, C):
    """
    Calculate the angle at point B given three points A, B, C in 2D space.
    """
    # Convert points to numpy arrays
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # Vectors BA and BC
    BA = A - B
    BC = C - B
    
    # Check if either vector is a zero vector
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)
    
    if norm_BA == 0 or norm_BC == 0:
        return np.nan  # Return NaN if we cannot compute the angle due to zero vectors
    
    # Calculate the cosine of the angle using dot product and magnitudes
    cosine_angle = np.dot(BA, BC) / (norm_BA * norm_BC)
    
    # Clamp cosine_angle to avoid rounding errors outside the range [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle_rad = np.arccos(cosine_angle)
    
    return angle_rad


    """
    Scale the 2D body pose coordinates to human space, assuming a known height.

    Args:
    joints_2d: (N, 2) array of 2D joint coordinates.
    person_height: The real-world height of the person (in meters).

    Returns:
    scaled_joints_2d: (N, 2) array of 2D joint coordinates scaled to human space.
    """
    # Find the top-most and bottom-most y-coordinates (assuming y is the vertical axis)
    top_y = joints_2d[34][1]  # Top of the person (e.g., head)
    bottom_y = np.min(joints_2d[:, 1])  # Bottom of the person (e.g., feet)

    # Calculate the current height of the person in 2D space
    current_height = np.abs(top_y - bottom_y)

    if current_height == 0:
        raise ValueError("Current height is zero. Ensure valid joint coordinates.")
    
    # Calculate the scaling factor to bring the height to person_height meters
    scale_factor = person_height / current_height

    # Apply the scaling factor to all coordinates
    scaled_joints_2d = joints_2d * scale_factor
    
    return scaled_joints_2d


def bring_coordinates_to_human_space_3d(joints_3d, person_height=1.90):
    """
    Scale the 3D body pose coordinates to human space, assuming a known height.
    
    Args:
    joints_3d: (N, 3) array of 3D joint coordinates.
    person_height: The real-world height of the person (in meters).
    
    Returns:
    scaled_joints_3d: (N, 3) array of 3D joint coordinates scaled to human space.
    """

    wrong_inference = False
    # Find the top-most and bottom-most y-coordinates (assuming y is up)
    top_z = joints_3d[34][2]
    bottom_z = np.min(joints_3d[:, 2])  # Bottom of the person (e.g., feet)

    
    # Calculate the current height of the person in 3D space
    current_height = np.abs(top_z - bottom_z)

    if current_height < 0.4:
        print("wrong inference with height = ",current_height)
        wrong_inference = True
        return None , wrong_inference
    
    # Calculate the scaling factor to bring the height to person_height meters
    scale_factor = person_height / current_height

    #print("Scale factor:", scale_factor)
    
    # Apply the scaling factor to all coordinates
    scaled_joints_3d = joints_3d * scale_factor
    
    return scaled_joints_3d, wrong_inference

def bring_coordinates_to_human_space_2d(joints_2d, person_height=1.90, pelvis_idx=0):
    """
    Scale the 2D body pose coordinates to human space and center around the pelvis.

    Args:
    joints_2d: (N, 2) array of 2D joint coordinates.
    person_height: The real-world height of the person (in meters).
    pelvis_idx: The index of the pelvis joint in the array.

    Returns:
    scaled_centered_joints_2d: (N, 2) array of 2D joint coordinates scaled and centered to human space.
    """

    # Center the joints around the pelvis
    pelvis_position = joints_2d[pelvis_idx]  # Get the pelvis coordinates
    scaled_centered_joints_2d = joints_2d - pelvis_position  # Subtract pelvis to center

    # Find the top-most and bottom-most y-coordinates (assuming y is the vertical axis)
    top_y = scaled_centered_joints_2d[34][1]  # Top of the person (e.g., head)
    bottom_y = np.max(scaled_centered_joints_2d[:, 1])  # Bottom of the person (e.g., feet)

    if bottom_y > top_y:
        current_height = np.abs(bottom_y - top_y)
    else:
        current_height = np.abs(top_y - bottom_y)
    

    if current_height == 0:
        raise ValueError("Current height is zero. Ensure valid joint coordinates.")
    
    # Calculate the scaling factor to bring the height to person_height meters
    scale_factor = person_height / current_height

    # Apply the scaling factor to all coordinates
    scaled_centered_joints_2d = scaled_centered_joints_2d * scale_factor

    

    return scaled_centered_joints_2d


def adjust_using_previous_vector(current_ankle_pos, previous_ankle_pos, current_joint_pos, previous_joint_pos, threshold):
    # Calculer la distance actuelle entre la cheville et le joint
    distance = np.linalg.norm(current_joint_pos - current_ankle_pos)
    
    # Si la distance est trop grande ou trop petite, ajuster en utilisant le vecteur précédent
    if distance > threshold:
        # Calculer le vecteur de la frame précédente
        previous_vector = previous_joint_pos - previous_ankle_pos
        
        # Appliquer ce vecteur à la nouvelle position de la cheville
        adjusted_pos = current_ankle_pos + previous_vector
        return adjusted_pos
    return current_joint_pos  # Sinon, ne pas ajuster



def calculate_features(player_id_frame,frame, match_data, action_name):
    """
    Placeholder function to calculate joint angles, angular velocity/acceleration, 
    and linear velocity/acceleration for the pelvis.
    
    You will need to implement your specific math for joint calculations here.
    """

    path_data = '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/aandre/datasets/Input_Model/'+match_data + '/'+action_name + '/frame_' + str(frame) +'/npy/'
    j3d_frame = np.load(path_data + 'j3d.npy')
    j2d_frame = np.load(path_data + 'j2d.npy')

    jump_class = np.load(path_data + 'all_jump_cls.npy')
    
    img_path = path_data + 'img_paths.pkl'

    df_path = pd.read_pickle(img_path)


    # Extract frame numbers from the path strings
    frame_numbers = [int(path.split('_')[-1].split('.')[0]) for path in df_path]

    # Create a DataFrame with the index and corresponding frame number
    df_with_frames = pd.DataFrame({
        'index_frame': frame_numbers
    })


    index_pose = df_with_frames[df_with_frames['index_frame'] == player_id_frame].index[0]

    j3d = j3d_frame[index_pose]
    j2d = j2d_frame[index_pose]
    jump_class_player = jump_class[index_pose]


    trans_mat = np.array([[1., 0, 0],
                      [0, 0, -1.],
                      [0, 1, 0]])


    j3d = np.dot(trans_mat,j3d.T).T
    j3d_norm = normalize_poses_orientationless(j3d, 4, 1, 0)

    # Scale the coordinates to a height of 1.90 meters
    j3d_human, wrong_inference = bring_coordinates_to_human_space_3d(j3d_norm, person_height=1.90)

    if wrong_inference:
        return None, None, None, None, None, None, None, None, None, None, None, None, None ,wrong_inference

    j2d_norm_human = bring_coordinates_to_human_space_2d(j2d, person_height=1.90, pelvis_idx=0)

    hip_angle_left_2D = round(calculate_angle_2d(j2d_norm_human[5], j2d_norm_human[4], j2d_norm_human[0]), 2)
    knee_angle_left_2D = round(calculate_angle_2d(j2d_norm_human[6], j2d_norm_human[5], j2d_norm_human[4]), 2)
    ankle_angle_left_2D = round(calculate_angle_2d(j2d_norm_human[30], j2d_norm_human[6], j2d_norm_human[5]), 2)

    hip_angle_right_2D = round(calculate_angle_2d(j2d_norm_human[2], j2d_norm_human[1], j2d_norm_human[0]), 2)
    knee_angle_right_2D = round(calculate_angle_2d(j2d_norm_human[3], j2d_norm_human[2], j2d_norm_human[1]), 2)
    ankle_angle_right_2D = round(calculate_angle_2d(j2d_norm_human[21], j2d_norm_human[3], j2d_norm_human[2]), 2)

        
    joint_angles_2D = {
        "right_leg": {
            "ankle": ankle_angle_right_2D,
            "knee": knee_angle_right_2D,
            "hip": hip_angle_right_2D
        },
        "left_leg": {
            "ankle": ankle_angle_left_2D,
            "knee": knee_angle_left_2D,
            "hip": hip_angle_left_2D
        }
    }


    angular_velocity_2D = {
        "right_leg": {
            "ankle": round(float(0), 2),
            "knee": round(float(0), 2),
            "hip": round(float(0), 2)
        },
        "left_leg": {
            "ankle": round(float(0), 2),
            "knee": round(float(0), 2),
            "hip": round(float(0), 2)
        }
    }

    angular_acceleration_2D = {
        "right_leg": {
            "ankle": round(float(0), 2),
            "knee": round(float(0), 2),
            "hip": round(float(0), 2)
        },
        "left_leg": {
            "ankle": round(float(0), 2),
            "knee": round(float(0), 2),
            "hip": round(float(0), 2)
        }
    }

    joint_angles_3D = {
        "right_leg": {
            "ankle": round(float(0), 2),
            "knee": round(float(0), 2),
            "hip": round(float(0), 2)
        },
        "left_leg": {
            "ankle": round(float(0), 2),
            "knee": round(float(0), 2),
            "hip": round(float(0), 2)
        }
    }

    angular_velocity_3D = {
        "right_leg": {
            "ankle": round(float(0), 2),
            "knee": round(float(0), 2),
            "hip": round(float(0), 2)
        },
        "left_leg": {
            "ankle": round(float(0), 2),
            "knee": round(float(0), 2),
            "hip": round(float(0), 2)
        }
    }

    angular_acceleration_3D = {
        "right_leg": {
            "ankle": round(float(0), 2),
            "knee": round(float(0), 2),
            "hip": round(float(0), 2)
        },
        "left_leg": {
            "ankle": round(float(0), 2),
            "knee": round(float(0), 2),
            "hip": round(float(0), 2)
        }
    }

    linear_position_3D = {
        "right_leg": {
            "ankle": np.round(np.array([0, 0, 0]), 2),
            "knee": np.round(np.array([0, 0, 0]), 2),
        },
        "left_leg": {
            "ankle": np.round(np.array([0, 0, 0]), 2),
            "knee": np.round(np.array([0, 0, 0]), 2),
        }
    }
    

    linear_velocity_3D = {
        "right_leg": {
            "ankle":  np.round(np.array([0, 0, 0]), 2),
            "knee":  np.round(np.array([0, 0, 0]), 2),
        },
        "left_leg": {
            "ankle":  np.round(np.array([0, 0, 0]), 2),
            "knee":  np.round(np.array([0, 0, 0]), 2),
        }
    }
    

    linear_acceleration_3D = {
        "right_leg": {
            "ankle":  np.round(np.array([0, 0, 0]), 2),
            "knee":  np.round(np.array([0, 0, 0]), 2),
        },
        "left_leg": {
            "ankle":  np.round(np.array([0, 0, 0]), 2),
            "knee":  np.round(np.array([0, 0, 0]), 2),
        }
    }

    j2d = {
        "coordinates": np.round(j2d_norm_human, 2)
    }

    j3d = {
        "coordinates": np.round(j3d_norm, 2)
    }

    j3d_human = {
        "coordinates": np.round(j3d_human, 2)
    }

    jump ={
        "class": np.float32(jump_class_player)
    }


    #return joint_angles_2D, angular_velocity_2D, angular_acceleration_2D, linear_position_2D, linear_velocity_2D, linear_acceleration_2D, joint_angles_3D, angular_velocity_3D, angular_acceleration_3D, linear_position_3D, linear_velocity_3D, linear_acceleration_3D, j2d, j3d, j3d_human, wrong_inference
    return joint_angles_2D, angular_velocity_2D, angular_acceleration_2D, joint_angles_3D, angular_velocity_3D, angular_acceleration_3D, linear_position_3D, linear_velocity_3D, linear_acceleration_3D, j2d, j3d, j3d_human, jump, wrong_inference

def process_clip_file(clip_file_path, match_data, action_name):
    players_data = defaultdict(lambda: {"player_id": None, "frames": []})
    count_players = {}

    with open(clip_file_path, 'r') as file:
        for i, line in enumerate(file):
            data = line.strip().split(',')
            
            frame_number = int(data[0])
            player_id = int(data[1])
            bbox_x = float(data[2])
            bbox_y = float(data[3])
            bbox_width = float(data[4])
            bbox_height = float(data[5])

            if frame_number in count_players.keys():
                count_players[frame_number] += 1
            else:
                count_players[frame_number] = 1

            if (bbox_x < 0) or (bbox_y < 0):
                continue

            
            nbr_players_frame = 0

            player_data = {
                "frame_number": frame_number,
                "bbox": {
                    "x": bbox_x,
                    "y": bbox_y,
                    "width": bbox_width,
                    "height": bbox_height
                },
                "nbr_players_frame": nbr_players_frame
            }
            
            # Calculate joint angles, velocities, accelerations
            #ang_2D, a_vel_2D, a_acc_2D, l_pos_2D, l_vel_2D, l_acc_2D, ang_3D, a_vel_3D, a_acc_3D, l_pos_3D, l_vel_3D, l_acc_3D, j2d, j3d, j3d_human, wrong_inference= calculate_features((count_players[frame_number] -1), frame_number, match_data, action_name)
            ang_2D, a_vel_2D, a_acc_2D, ang_3D, a_vel_3D, a_acc_3D, l_pos_3D, l_vel_3D, l_acc_3D, j2d, j3d, j3d_human, jump, wrong_inference= calculate_features((count_players[frame_number] -1), frame_number, match_data, action_name)
            
            if wrong_inference:
                continue

            # Add calculations to the player's frame data
            player_data["joint_angles_2D"] = ang_2D
            player_data["angular_velocity_2D"] = a_vel_2D
            player_data["angular_acceleration_2D"] = a_acc_2D

            player_data["joint_angles_3D"] = ang_3D
            player_data["angular_velocity_3D"] = a_vel_3D
            player_data["angular_acceleration_3D"] = a_acc_3D

            player_data["linear_position_3D"] = l_pos_3D
            player_data["linear_velocity_3D"] = l_vel_3D
            player_data["linear_acceleration_3D"] = l_acc_3D

            player_data["j2d"] = j2d

            player_data["j3d_human"] = j3d_human

            player_data["jump"] = float(jump['class'])

            # Store the data for the player
            players_data[player_id]["player_id"] = player_id
            players_data[player_id]["frames"].append(player_data)

    for player_info in players_data.values():
        for frame in player_info["frames"]:
            frame_number = frame["frame_number"]
            total_players = count_players.get(frame_number, 1)  # Get total players in the frame
            frame["nbr_players_frame"] = total_players

    
    players_data = {
    "clip": action_name,
    "match": match_data,
    "players": {f"player_{player_id}": player_info for player_id, player_info in players_data.items()}
    }
    
    return players_data


def save_to_json(data, output_file):
    # Define a custom encoder for handling NumPy arrays
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert ndarray to list
            return json.JSONEncoder.default(self, obj)

    with open(output_file, 'w') as json_file:
        # Use the custom encoder to convert NumPy arrays to lists
        json.dump({"action_data": data}, json_file, indent=4, cls=NumpyEncoder)

def replace_nan_with_last_valid_or_default(angle_values, default_value, prev_valid_value):
    """
    Replace NaN values in the angle array. If NaN is present:
    - Replace it with the last valid value (prev_valid_value).
    - If no valid value is found within a specified range, use the default value.

    Args:
    angle_values: The current frame's angle values.
    default_value: Default angle value to use if no valid frame is found.
    prev_valid_value: The last valid angle value from the previous frames.

    Returns:
    corrected_angle: The angle value without NaN, replaced with either the last valid or default value.
    """
    if np.isnan(angle_values):
        if prev_valid_value is not None:
            return prev_valid_value  # Use the last valid value
        else:
            return default_value  # No valid values found, use default
    return angle_values



def calculate_angles(frames):

    for i, frame in enumerate(frames):
        hip_angle_left_3D = round(calculate_angle_3d(frame['j3d_human']['coordinates'][5], frame['j3d_human']['coordinates'][4], frame['j3d_human']['coordinates'][0]), 2)
        knee_angle_left_3D = round(calculate_angle_3d(frame['j3d_human']['coordinates'][6], frame['j3d_human']['coordinates'][5], frame['j3d_human']['coordinates'][4]), 2)
        ankle_angle_left_3D = round(calculate_angle_3d(frame['j3d_human']['coordinates'][30], frame['j3d_human']['coordinates'][6], frame['j3d_human']['coordinates'][5]), 2)

        hip_angle_right_3D = round(calculate_angle_3d(frame['j3d_human']['coordinates'][2], frame['j3d_human']['coordinates'][1], frame['j3d_human']['coordinates'][0]), 2)
        knee_angle_right_3D = round(calculate_angle_3d(frame['j3d_human']['coordinates'][3], frame['j3d_human']['coordinates'][2], frame['j3d_human']['coordinates'][1]), 2)
        ankle_angle_right_3D = round(calculate_angle_3d(frame['j3d_human']['coordinates'][21], frame['j3d_human']['coordinates'][3], frame['j3d_human']['coordinates'][2]), 2)


        joint_angles_3D = {
            "right_leg": {
                "ankle": ankle_angle_right_3D,
                "knee": knee_angle_right_3D,
                "hip": hip_angle_right_3D
            },
            "left_leg": {
                "ankle": ankle_angle_left_3D,
                "knee": knee_angle_left_3D,
                "hip": hip_angle_left_3D
            }
        }

        linear_position_3D = {
            "right_leg": {
                "ankle": np.round(frame['j3d_human']['coordinates'][3], 2),
                "knee": np.round(frame['j3d_human']['coordinates'][2], 2),
            },
            "left_leg": {
                "ankle": np.round(frame['j3d_human']['coordinates'][6], 2),
                "knee": np.round(frame['j3d_human']['coordinates'][5], 2),
            }
        }

        frames[i]["joint_angles_3D"] =joint_angles_3D
        frames[i]["linear_position_3D"] =linear_position_3D

    return frames


def compute_velocities_and_accelerations(frames, frame_rate, frame_gap=5):
    """
    Compute the linear and angular velocities and accelerations between frames,
    comparing them with a difference of `frame_gap` frames, except for the first 
    few frames where the first frame is used as the reference.
    
    Also, handles NaN angle values by replacing them with previous values or default values if needed.
    
    Args:
    frames: List of frames containing player pose data.
    frame_rate: Frame rate of the data (frames per second).
    frame_gap: The number of frames to skip when computing velocities and accelerations.
    
    Returns:
    updated_frames: List of frames with updated velocity and acceleration data.
    """
    new_starting_frame = False
    index_start = 0
    last_valid_angles_2D = {'right_leg': {'ankle': None, 'knee': None, 'hip': None}, 'left_leg': {'ankle': None, 'knee': None, 'hip': None}}
    last_valid_angles_3D = {'right_leg': {'ankle': None, 'knee': None, 'hip': None}, 'left_leg': {'ankle': None, 'knee': None, 'hip': None}}

    # Check NaN for the first frame
    for leg in ['right_leg', 'left_leg']:
        for joint in ['hip', 'knee', 'ankle']:
            # 2D angles
            curr_angle_2D = frames[0]['joint_angles_2D'][leg][joint]
            default_value_2D = round(float((np.pi/2)+0.1), 2) if joint == 'ankle' else round(float(np.pi-0.6), 2) if joint == 'knee' else round(float((np.pi/2)+0.1), 2)
            corrected_angle_2D = replace_nan_with_last_valid_or_default(
                curr_angle_2D,
                default_value=default_value_2D,
                prev_valid_value=default_value_2D
            )
            frames[0]['joint_angles_2D'][leg][joint] = corrected_angle_2D

            # 3D angles
            curr_angle_3D = frames[0]['joint_angles_3D'][leg][joint]
            default_value_3D = round(float((np.pi/2)+0.1), 2) if joint == 'ankle' else round(float(np.pi-0.6), 2) if joint == 'knee' else round(float((np.pi/2)+0.1), 2)
            corrected_angle_3D = replace_nan_with_last_valid_or_default(
                curr_angle_3D,
                default_value=default_value_3D,
                prev_valid_value=default_value_3D
            )
            frames[0]['joint_angles_3D'][leg][joint] = corrected_angle_3D

            if not np.isnan(corrected_angle_2D):
                last_valid_angles_2D[leg][joint] = corrected_angle_2D

            if not np.isnan(corrected_angle_3D):
                last_valid_angles_3D[leg][joint] = corrected_angle_3D

    # Iterate through remaining frames
    for i in range(1, len(frames)):
        if new_starting_frame:
            if (i - index_start) < frame_gap:
                prev_frame = frames[index_start]
            else:
                new_starting_frame = False
                prev_frame = frames[i - frame_gap]
        else:
            if i < frame_gap:
                prev_frame = frames[0]  # Use the first frame as reference for the first few frames
            else:
                prev_frame = frames[i - frame_gap]  # Use the frame `i - frame_gap` as the reference frame
        
        curr_frame = frames[i]

        # Calculate the time difference based on frame numbers
        frame_diff = curr_frame['frame_number'] - prev_frame['frame_number']
        if frame_diff > 15:
            new_starting_frame = True
            index_start = i
            for leg in ['right_leg', 'left_leg']:
                for joint in ['hip', 'knee', 'ankle']:
                    prev_angle_2D = prev_frame['joint_angles_2D'][leg][joint]
                    curr_angle_2D = curr_frame['joint_angles_2D'][leg][joint]
                    prev_angle_3D = prev_frame['joint_angles_3D'][leg][joint]
                    curr_angle_3D = curr_frame['joint_angles_3D'][leg][joint]

                    # Check if the current 2D angle is NaN and apply replacement strategy
                    default_value_2D = round(float((np.pi/2)+0.1), 2) if joint == 'ankle' else round(float(np.pi-0.6), 2) if joint == 'knee' else round(float((np.pi/2)+0.1), 2)
                    corrected_angle_2D = replace_nan_with_last_valid_or_default(
                        curr_angle_2D,
                        default_value=default_value_2D,
                        prev_valid_value=last_valid_angles_2D[leg][joint]
                    )
                    curr_frame['joint_angles_2D'][leg][joint] = corrected_angle_2D

                    # Check if the current 3D angle is NaN and apply replacement strategy
                    default_value_3D = round(float((np.pi/2)+0.1), 2) if joint == 'ankle' else round(float(np.pi-0.6), 2) if joint == 'knee' else round(float((np.pi/2)+0.1), 2)
                    corrected_angle_3D = replace_nan_with_last_valid_or_default(
                        curr_angle_3D,
                        default_value=default_value_3D,
                        prev_valid_value=last_valid_angles_3D[leg][joint]
                    )
                    curr_frame['joint_angles_3D'][leg][joint] = corrected_angle_3D

                    if not np.isnan(corrected_angle_2D):
                        last_valid_angles_2D[leg][joint] = corrected_angle_2D

                    if not np.isnan(corrected_angle_3D):
                        last_valid_angles_3D[leg][joint] = corrected_angle_3D

        dt = frame_diff / frame_rate 

        if dt == 0:
            continue 

        # List of body parts for which we want to calculate velocity and acceleration
        body_parts = ['right_leg', 'left_leg']
        joints = ['ankle', 'knee']

        # Loop through body parts and joints to calculate linear velocity and acceleration
        for part in body_parts:
            for joint in joints:
                # Previous and current positions
                prev_joint_3D = np.array(prev_frame['linear_position_3D'][part][joint])
                curr_joint_3D = np.array(curr_frame['linear_position_3D'][part][joint])

                # Linear velocity (difference in position / time)
                linear_velocity_3D = (curr_joint_3D - prev_joint_3D) / dt

                # Previous velocity
                prev_velocity_3D = np.array(prev_frame['linear_velocity_3D'][part][joint])

                # Linear acceleration (difference in velocity / time)
                linear_acceleration_3D = (linear_velocity_3D - prev_velocity_3D) / dt

                # Update current frame's linear velocity and acceleration
                curr_frame['linear_velocity_3D'][part][joint] = np.round(linear_velocity_3D, 2).tolist()
                curr_frame['linear_acceleration_3D'][part][joint] = np.round(linear_acceleration_3D, 2).tolist()

        # Calculate angular velocity and acceleration for joints (2D and 3D)
        for leg in ['right_leg', 'left_leg']:
            # Angular velocity (difference in angles / time)
            angular_velocity_2D = {}
            angular_velocity_3D = {}

            for joint in ['hip', 'knee', 'ankle']:
                prev_angle_2D = prev_frame['joint_angles_2D'][leg][joint]
                curr_angle_2D = curr_frame['joint_angles_2D'][leg][joint]

                prev_angle_3D = prev_frame['joint_angles_3D'][leg][joint]
                curr_angle_3D = curr_frame['joint_angles_3D'][leg][joint]

                # Check if the current 2D angle is NaN and apply replacement strategy
                default_value_2D = round(float((np.pi/2)+0.1), 2) if joint == 'ankle' else round(float(np.pi-0.6), 2) if joint == 'knee' else round(float((np.pi/2)+0.1), 2)
                corrected_angle_2D = replace_nan_with_last_valid_or_default(
                    curr_angle_2D,
                    default_value=default_value_2D,
                    prev_valid_value=last_valid_angles_2D[leg][joint]
                )
                curr_frame['joint_angles_2D'][leg][joint] = corrected_angle_2D

                # Check if the current 3D angle is NaN and apply replacement strategy
                default_value_3D = round(float((np.pi/2)+0.1), 2) if joint == 'ankle' else round(float(np.pi-0.6), 2) if joint == 'knee' else round(float((np.pi/2)+0.1), 2)
                corrected_angle_3D = replace_nan_with_last_valid_or_default(
                    curr_angle_3D,
                    default_value=default_value_3D,
                    prev_valid_value=last_valid_angles_3D[leg][joint]
                )
                curr_frame['joint_angles_3D'][leg][joint] = corrected_angle_3D

                # Update the last valid angle if it's a valid number
                if not np.isnan(corrected_angle_2D):
                    last_valid_angles_2D[leg][joint] = corrected_angle_2D

                if not np.isnan(corrected_angle_3D):
                    last_valid_angles_3D[leg][joint] = corrected_angle_3D

                # Angular velocity
                angular_velocity_2D[joint] = (corrected_angle_2D - prev_angle_2D) / dt if not np.isnan(prev_angle_2D) else 0
                angular_velocity_3D[joint] = (corrected_angle_3D - prev_angle_3D) / dt

                # Angular acceleration (difference in angular velocity / time)
                prev_angular_velocity_2D = prev_frame['angular_velocity_2D'][leg][joint]
                prev_angular_velocity_3D = prev_frame['angular_velocity_3D'][leg][joint]

                angular_acceleration_2D = (angular_velocity_2D[joint] - prev_angular_velocity_2D) / dt
                angular_acceleration_3D = (angular_velocity_3D[joint] - prev_angular_velocity_3D) / dt

                # Update current frame's angular velocity and acceleration
                curr_frame['angular_velocity_2D'][leg][joint] = round(angular_velocity_2D[joint], 2)
                curr_frame['angular_acceleration_2D'][leg][joint] = round(angular_acceleration_2D, 2)

                curr_frame['angular_velocity_3D'][leg][joint] = round(angular_velocity_3D[joint], 2)
                curr_frame['angular_acceleration_3D'][leg][joint] = round(angular_acceleration_3D, 2)

    return frames



def get_video_frame_rate(video_path):
    """
    Returns the frame rate (frames per second) of the video file at the given path.
    
    Args:
    video_path: Path to the video file.
    
    Returns:
    frame_rate: The frame rate (fps) of the video.
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get the frame rate (fps)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    
    # Release the video capture object
    video.release()
    
    return frame_rate


def process_all_players(data, frame_rate, frame_gap=5):
    """
    Apply the compute_velocities_and_accelerations function to all players in the JSON data.

    Args:
    data: JSON-like dictionary containing player data.
    frame_rate: Frame rate of the video (fps).
    frame_gap: The number of frames to skip when computing velocities and accelerations.

    Returns:
    updated_data: JSON data with updated velocities and accelerations for all players.
    """
    # Loop through all players in the data
    for player_id, player_info in data['players'].items():
        # Get the frames for the player
        frames = player_info['frames']
        frames = calculate_angles(frames)
        # Compute velocities and accelerations for the player's frames
        updated_frames = compute_velocities_and_accelerations(frames, frame_rate, frame_gap)

        # Update the player's frames in the original data
        data['players'][player_id]['frames'] = updated_frames

    return data

def main():
    parser = argparse.ArgumentParser(description="Process clip file for basketball match data.")
    
    # Add the match name argument
    parser.add_argument('--match_name', type=str, default=None, help="The name of the match to process (e.g., MEM_LAK, PAC_ATL).")
    parser.add_argument('--injuries_data', type=bool, default=False, help="Injuries data or not.")
    args = parser.parse_args()
    if args.match_name == None:
        match_names = ['MEM_LAK', 'PAC_ATL', 'POR_SAC', 'SAS_DET']
    else:
        match_names = [args.match_name]
    for match_name in match_names:
        directory_path= f'/n/home12/aandre/MixSortTracking/YOLOX_outputs/{match_name}'
        action_names = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
        if args.injuries_data == False:
            action_names = sorted(action_names, key=int)
        for action_name in action_names:
            print(f"Processing {match_name}/{str(action_name)}...")
            #clip_file_path = '/Users/strom/Desktop/clip.txt'  
            #output_json_path = '/Users/strom/Desktop/players_data.json' 
            #video_path = "/Users/strom/Desktop/CHI_NYK/530.mp4"

            clip_file_path= f'/n/home12/aandre/MixSortTracking/YOLOX_outputs/{match_name}/{str(action_name)}/track_results/clip.txt'
            output_json_path = f'/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/aandre/datasets/{match_name}/{str(action_name)}/' 
            video_path = f"/n/home12/aandre/MixSortTracking/nba_videos/{match_name}/{str(action_name)}.mp4"

            os.makedirs(output_json_path, exist_ok=True)

            output_json_path = output_json_path + 'players_data.json'
            
            players_data = process_clip_file(clip_file_path, match_name, str(action_name))


            frame_rate = get_video_frame_rate(video_path)
            print(f"Frame rate of the video: {frame_rate} fps")

            # Compute velocities and accelerations
            updated_data = process_all_players(players_data, frame_rate, frame_gap=5)
            save_to_json(updated_data, output_json_path)


if __name__ == "__main__":
    main()