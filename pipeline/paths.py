"""Configurable paths for the NBA / custom-video pipeline.

Override any value with environment variables (see README).
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

YOLOX_OUTPUTS = os.environ.get("MIXSORT_YOLOX_OUTPUTS", os.path.join(REPO_ROOT, "YOLOX_outputs"))
DATASETS_ROOT = os.environ.get("MIXSORT_DATASETS", os.path.join(REPO_ROOT, "datasets"))
NBA_VIDEOS = os.environ.get("MIXSORT_NBA_VIDEOS", os.path.join(REPO_ROOT, "nba_videos"))
POSE_INPUT_ROOT = os.environ.get("MIXSORT_POSE_INPUT", os.path.join(DATASETS_ROOT, "Input_Model"))
OUTPUT_DIR = os.environ.get("MIXSORT_OUTPUT", os.path.join(REPO_ROOT, "output"))


def yolox_match_dir(match_name: str) -> str:
    return os.path.join(YOLOX_OUTPUTS, match_name)


def dataset_action_dir(match_name: str, action_name: str) -> str:
    return os.path.join(DATASETS_ROOT, match_name, str(action_name))


def clip_txt_path(match_name: str, action_name: str) -> str:
    return os.path.join(
        yolox_match_dir(match_name), str(action_name), "track_results", "clip.txt"
    )


def nba_video_path(match_name: str, action_name: str) -> str:
    return os.path.join(NBA_VIDEOS, match_name, f"{action_name}.mp4")


def pose_npy_dir(match_name: str, action_name: str, frame: int) -> str:
    return os.path.join(POSE_INPUT_ROOT, match_name, action_name, f"frame_{frame}", "npy")
