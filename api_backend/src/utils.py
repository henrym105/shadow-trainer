import cv2 
import json
import logging
import os
import pickle
import random
from typing import IO, Any

import boto3
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] in %(name)s.%(funcName)s() --> %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# YAML/Configuration utilities
# =============================================================================

def construct_include(loader: 'Loader', node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json',):
            return json.load(f)
        else:
            return ''.join(f.readlines())


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config


# =============================================================================
# Mathematical/geometric utilities
# =============================================================================

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2 or X.shape[-1] == 3
    result = np.copy(X)
    result[..., :2] = X[..., :2] / w * 2 - [1, h / w]
    return result


def qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


# =============================================================================
# Tensor/array utilities
# =============================================================================

def wrap(func, *args, unsqueeze=False):
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
    result = func(*args)
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


# =============================================================================
# File system utilities
# =============================================================================

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content


def download_file_if_not_exists(filename_pattern, local_dir, s3_bucket="shadow-trainer-prod", s3_prefix="model_weights") -> str:
    """Checks if a file matching filename_pattern exists in local_dir. 
    If not, searches and downloads from S3.
    Returns the local file path of the first match found.
    """
    import fnmatch
    os.makedirs(local_dir, exist_ok=True)
    # Search locally
    local_matches = fnmatch.filter(os.listdir(local_dir), filename_pattern)
    if local_matches:
        local_path = os.path.join(local_dir, local_matches[0])
        print(f"[INFO] Found checkpoint locally: {local_path}")
        return local_path

    # Search S3
    s3 = boto3.client("s3")
    s3_prefix_full = f"{s3_prefix}/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix_full):
        if "Contents" in page:
            for obj in page["Contents"]:
                s3_key = obj["Key"]
                s3_filename = os.path.basename(s3_key)
                if fnmatch.fnmatch(s3_filename, filename_pattern):
                    local_path = os.path.join(local_dir, s3_filename)
                    print(f"[INFO] Downloading model weights from s3://{s3_bucket}/{s3_key} to {local_path}")
                    s3.download_file(s3_bucket, s3_key, local_path)
                    return local_path

    raise FileNotFoundError(f"No checkpoint found matching pattern '{filename_pattern}' in {local_dir} or s3://{s3_bucket}/{s3_prefix}/")


def get_pytorch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_frame_info(frame: np.ndarray) -> dict:
    """Get the joint names and their corresponding coordinates from a single frame of 3D key points.

    Args:
        frame (np.ndarray): A single frame of 3D key points, expected shape (
            17, 3), where each row corresponds to a joint and each column corresponds to x, y, z coordinates.

    Returns:
        dict: A dictionary mapping joint names to their coordinates. Like {'Hip': [x, y, z], 'Right Hip': [x, y, z], ...}
    """
    assert frame.shape == (17, 3), f"Expected frame shape (17, 3), got {frame.shape}. Ensure the input is a single frame of 3D key points."
    joint_names = [
        "Hip", "Right Hip", "Right Knee", "Right Ankle",
        "Left Hip", "Left Knee", "Left Ankle", "Spine",
        "Thorax", "Neck", "Head", "Left Shoulder",
        "Left Elbow", "Left Wrist", "Right Shoulder",
        "Right Elbow", "Right Wrist"
        ]
    joints = {joint_names[i]: frame[i] for i in range(len(frame))}
    return joints


def get_frame_size(cap: cv2.VideoCapture) -> tuple:
    """Get the size of the first frame in the video capture.
    
    Args:
        cap (cv2.VideoCapture): OpenCV video capture object.

    Returns:
        tuple: Size of the first frame as (height, width).
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (height, width, 3)  # Assuming 3 channels (RGB)



# =============================================================================
# Development/debugging utilities
# =============================================================================

def print_args(args):
    print("[INFO] Input arguments:")
    for key, val in args.items():
        print(f"[INFO]   {key}: {val}")
        

def set_random_seed(seed):
    """Sets random seed for training reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_param_numbers(model):
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    return model_params
