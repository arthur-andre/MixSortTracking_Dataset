import os
import numpy as np
import json
import cv2
from tqdm import tqdm
import shutil
import argparse
import glob

def extract_and_save_frames(video_path, base_output_folder):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_number = 0
    im_width = 0
    im_height = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Extract frame rate from video metadata

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number == 0:
            # Get dimensions from the first frame
            im_height, im_width = frame.shape[:2]

        # Ensure the base output folder exists
        os.makedirs(base_output_folder, exist_ok=True)

        # Construct the filename for each frame
        frame_file = os.path.join(base_output_folder, f"{frame_number+1:06d}.jpg")

        # Save the frame with specified JPEG quality
        cv2.imwrite(frame_file, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Increment the frame counter
        frame_number += 1

    # Release the video capture object
    cap.release()

    folder_parts = base_output_folder.split("/")
    #remove last string of the folder parts
    sequence_path = base_output_folder.replace(folder_parts[-1], '')
    if len(folder_parts) > 1:
        sequence_name = folder_parts[-2]
    else:
        sequence_name = "DefaultSequenceName"
    # After extracting frames, create or update the seqinfo.ini file

    seqinfo_path = os.path.join(sequence_path, 'seqinfo.ini')
    with open(seqinfo_path, 'w') as f:
        f.write("[Sequence]\n")
        f.write(f"name={sequence_name}\n")  
        f.write("imDir=img1\n")
        f.write(f"frameRate={frame_rate:.2f}\n")
        f.write(f"seqLength={frame_number}\n")  
        f.write(f"imWidth={im_width}\n")
        f.write(f"imHeight={im_height}\n")
        f.write("imExt=.jpg\n")

    print(f"Extracted {frame_number} frames to '{base_output_folder}'.")
    print(f"Sequence information saved to '{seqinfo_path}'.")

def convert_coco(base_output_folder, output_folder, split):
    data_path = os.path.join(base_output_folder, split)
    out_path = os.path.join(output_folder, "{}.json".format(split))

    out = {
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": [{
            "id": 1,
            "name": "pedestrian"
        }]
    }
    video_list = os.listdir(data_path)
    image_cnt = 0
    video_cnt = 0
    for seq in tqdm(sorted(video_list)):
        if ".DS_Store" in seq:
            continue
        video_cnt += 1  
        out["videos"].append({"id": video_cnt, "file_name": seq})

        seq_path = os.path.join(data_path, seq)
        img_path = os.path.join(seq_path, "img1")

        images = os.listdir(img_path)
        num_images = len([image for image in images
                          if "jpg" in image])  
        
        info_path = os.path.join(seq_path, "seqinfo.ini")
        with open(info_path, "r") as f:
            for line in f:
                if 'imWidth' in line:
                    width = int(line.split('=')[1])
                elif 'imHeight' in line:
                    height = int(line.split('=')[1])


        image_range = [0, num_images - 1]

        for i in range(num_images):
            if i < image_range[0] or i > image_range[1]:
                continue
            image_info = {
                "file_name": "{}/img1/{:06d}.jpg".format(seq,
                                                         i + 1),  # image name.
                "id":
                image_cnt + i + 1,  # image number in the entire training set.
                "frame_id": i + 1 - image_range[
                    0],  # image number in the video sequence, starting from 1.
                "prev_image_id": image_cnt +
                i if i > 0 else -1,  # image number in the entire training set.
                "next_image_id":
                image_cnt + i + 2 if i < num_images - 1 else -1,
                "video_id": video_cnt,
                "height": height,
                "width": width
            }
            out["images"].append(image_info)
        print("{}: {} images".format(seq, num_images))

        image_cnt += num_images
    print("loaded {} for {} images and {} samples".format(
        split, len(out["images"]), len(out["annotations"])))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

def convert_coco_windows(base_output_folder, output_folder, split):
    data_path = base_output_folder + '/' + split
    out_path = os.path.join(output_folder, "{}.json".format(split))

    out = {
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": [{
            "id": 1,
            "name": "pedestrian"
        }]
    }
    video_list = os.listdir(data_path)
    image_cnt = 0
    video_cnt = 0
    for seq in tqdm(sorted(video_list)):
        if ".DS_Store" in seq:
            continue
        video_cnt += 1  
        out["videos"].append({"id": video_cnt, "file_name": seq})

        seq_path = data_path + '/' + seq
        img_path = seq_path + '/' + "img1"

        images = os.listdir(img_path)
        num_images = len([image for image in images
                          if "jpg" in image])  
        
        info_path = os.path.join(seq_path, "seqinfo.ini")
        info_path = info_path.replace("\\", "/")

        with open(info_path, "r") as f:
            for line in f:
                if 'imWidth' in line:
                    width = int(line.split('=')[1])
                elif 'imHeight' in line:
                    height = int(line.split('=')[1])


        image_range = [0, num_images - 1]

        for i in range(num_images):
            if i < image_range[0] or i > image_range[1]:
                continue
            # img = cv2.imread(
            #     os.path.join(data_path,
            #                  "{}/img1/{:06d}.jpg".format(seq, i + 1)))
            # height, width = img.shape[:2]
            image_info = {
                "file_name": "{}/img1/{:06d}.jpg".format(seq,
                                                         i + 1),  # image name.
                "id":
                image_cnt + i + 1,  # image number in the entire training set.
                "frame_id": i + 1 - image_range[
                    0],  # image number in the video sequence, starting from 1.
                "prev_image_id": image_cnt +
                i if i > 0 else -1,  # image number in the entire training set.
                "next_image_id":
                image_cnt + i + 2 if i < num_images - 1 else -1,
                "video_id": video_cnt,
                "height": height,
                "width": width
            }
            out["images"].append(image_info)
        print("{}: {} images".format(seq, num_images))

        image_cnt += num_images
    print("loaded {} for {} images and {} samples".format(
        split, len(out["images"]), len(out["annotations"])))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

def process_videos(match_selection, sequence_name, args):

    output_folder = f"/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/aandre/datasets/{match_selection}"

    #output_folder = f"datasets/{match_selection}"   #### CHANGEEEEEEEEEEEEEE

    # Define the input folder path
    input_folder = f"nba_videos/{match_selection}"

    # Find all .mp4 files in the input folder
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))

    if not video_files:
        print(f"No .mp4 files found in {input_folder}.")
        return
    else:
        print(len(video_files)," videos found.")

    video_files = sorted(video_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # Extract the number from the filename (without the .mp4)
    numbers_sorted = sorted([int(os.path.splitext(os.path.basename(x))[0]) for x in video_files])

    numbers_set = set(numbers_sorted)

    # Step 3: Save the sorted numbers to a JSON file
    output_file = output_folder+'/sorted_actions.json'
    with open(output_file, 'w') as f:
        json.dump(list(numbers_set), f)

    # Iterate through each video file
    for video_file in video_files:
        # Extract video name without extension
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        
        # Define the output path for frames
        output_frames = os.path.join(output_folder, video_name, 'test', sequence_name, 'img1')
        os.makedirs(output_frames, exist_ok=True)
        
        # Extract and save frames for the current video
        print(f"Extracting frames from {video_file} to {output_frames}")
        extract_and_save_frames(video_file, output_frames)

        # Define the path for annotations
        out_path_json = os.path.join(output_folder, video_name, "annotations")
        os.makedirs(out_path_json, exist_ok=True)
        
        # Set the split to 'test' for now (can be adjusted)
        split = "test"
        out_based_json = os.path.join(output_folder, video_name)
        # Convert the extracted frames to COCO format
        print(f"Converting frames to COCO format and saving to {out_path_json}")
        convert_coco(out_based_json, out_path_json, split)

        print(f"Processed video: {video_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video and save them as JPG files.")
    parser.add_argument("match_selection", help="Match that we want to process (name of the folder containing videos).")
    parser.add_argument("sequence_name", help="Name of the sequence for output organization.")
    args = parser.parse_args()

    # Create the main output folder for the match
    folder_path = f"/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/aandre/datasets/{args.match_selection}"
    #folder_path = f"datasets/{args.match_selection}"               #### CHANGEEEEEEEEEEEEEE
    os.makedirs(folder_path, exist_ok=True)

    # Process all .mp4 videos in the match_selection folder
    process_videos(args.match_selection, args.sequence_name, args)



