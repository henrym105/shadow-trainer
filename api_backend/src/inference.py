import argparse
import copy
import glob
import logging
import shutil
import os
from os.path import join as pjoin

import cv2
import matplotlib
import matplotlib.axis
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pprint import pprint
from tqdm import tqdm

from src.preprocess import h36m_coco_format
from src.hrnet.gen_kpts import gen_video_kpts
from src.utils import download_file_if_not_exists, get_frame_info, normalize_screen_coordinates, camera_to_world, get_config, get_pytorch_device
from src.model.MotionAGFormer import MotionAGFormer
from src.visualizations import resample_pose_sequence, time_warp_pro_video
from src.yolo2d import YOLOPoseEstimator, rotate_video_until_upright

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] in %(name)s.%(funcName)s() --> %(message)s')
logger = logging.getLogger(__name__)


plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ---------------------- CONSTANTS ----------------------
BACKEND_API_DIR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = pjoin(BACKEND_API_DIR_ROOT, "checkpoint")
OUTPUT_FOLDER_RAW_KEYPOINTS = "raw_keypoints"
KEYPOINTS_FILE_2D = "2D_keypoints.npy"
KEYPOINTS_FILE_3D_USER = "user_3D_keypoints.npy"
KEYPOINTS_FILE_3D_PRO  = "pro_3D_keypoints.npy"

# debug flag
DEBUG = True
# -------------------------------------------------------

def get_joint_colors(color='R', use_0_255_range=False):
    """ Returns the left and right joint colors based on the specified color scheme.
    Args:
        color (str): Color scheme to use. Options are 'RB', 'R', 'G', 'blk'.
        use_0_255_range (bool): If True, returns colors in 0-255 range, otherwise in 0-1 range.

    Returns:
        tuple: Left and right joint colors as tuples of RGB values.
    """
    format_options = {
        'RB':  ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),     # Blue, Red
        'R':   ((1.0, 0.0, 0.0), (0.4, 0.0, 0.0)),     # Dark Red, Bright Red
        'G':   ((0.0, 0.6, 0.0), (0.0, 0.6, 0.0)),     # Dark Green, Bright Green
        'blk': ((0.6, 0.6, 0.6), (0.3, 0.3, 0.3)),     # Light gray, Dark gray
    }

    lcolor, rcolor = format_options.get(color, ((0,0,1),(1,0,0)))  # Default to Blue, Red

    if use_0_255_range:
        # Adjust for OpenCV's default color format of range 0-255 and BGR order
        lcolor = lcolor[::-1]
        rcolor = rcolor[::-1]
        lcolor = tuple(int(255 * c) for c in lcolor)
        rcolor = tuple(int(255 * c) for c in rcolor)

    return (lcolor, rcolor)


def show2Dpose(kps: np.ndarray, img: np.ndarray, color='R', is_lefty: bool = False) -> np.ndarray:
    """Draws a 2D human pose skeleton on the given image using the provided keypoints.

    Args:
        kps (np.ndarray): An array of shape (17, 3) containing the 2D coordinates and confidence scores (x, y, conf) for all 17 keypoints in this image. 
        img (np.ndarray): The image (as a NumPy array) on which to draw the skeleton.
        color (str): Color scheme to use for the skeleton.
        is_lefty (bool): If True, mark left foot (index 6) as front foot; else right foot (index 3).

    Returns:
        np.ndarray: The image with the 2D pose skeleton drawn on it.
    """
    # convert kps to int
    kps = kps.astype(int)

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                    [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                    [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    # 1 = left arm & leg, 0 = right arm/leg, torso, neck
    LR = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor, rcolor = get_joint_colors(color, use_0_255_range=True)
    thickness = 3
    assert kps.shape == (17,3), "Keypoints should be a 2D array with shape (n_person, n_frames, 17, 3). Received shape: {}".format(kps.shape)

    # Determine front foot keypoint index based on is_lefty
    front_foot_idx = 3 if is_lefty else 6
    green_color = (0, 255, 0)  # Green color in BGR format

    for j,c in enumerate(connections):
        # EXAMPLE: c = [5,6] --> connecting 5th keypoint (left knee) to 6th keypoint (left ankle)
        #          kps[c[1]] = kps[6] = (x,y) coordinated of the left ankle

        start = kps[c[0]]
        end = kps[c[1]]

        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 0, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 0, 0), radius=3)

    # Draw the front foot in green
    front_foot = kps[front_foot_idx]
    cv2.circle(img, (front_foot[0], front_foot[1]), thickness=-1, color=green_color, radius=5)

    return img


def show3Dpose(vals, ax, color='RB', camera_view_height = 15.0, camera_view_z_rotation = 70.0):
    # --------------------------------
    # use this to modify the camera perspective angle
    ax.view_init(elev=camera_view_height, azim=camera_view_z_rotation)
    # --------------------------------
    
    lcolor, rcolor = get_joint_colors(color)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in range(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.7
    RADIUS_Z = 0.7
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    show_axis_label = False
    ax.tick_params('x', labelbottom=show_axis_label)
    ax.tick_params('y', labelleft=show_axis_label)
    ax.tick_params('z', labelleft=show_axis_label)

    # Plot small black dots at each keypoint
    vals = np.asarray(vals)
    ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], c='k', s=3, depthshade=False, alpha=0.5)



def get_pose2D(video_path, output_file, device, yolo_version: str = "11") -> np.ndarray:
    """ Generate 2D pose keypoints from a video file and save them as a .npy file.
    
    Args:
        video_path (str): Path to the input video file.
        output_file (str): Path to save the output 2D keypoints .npy file.
        device (str): Device to run the model on ('cpu', 'cuda', or 'mps').

    Returns:
        str: Path to the saved 2D keypoints .npy file.
    """
    logger.info('\n\nGenerating 2D pose...')

    if yolo_version == "3":
        # Download hrnet and v3 weights from S3 if they don't exist locally
        download_file_if_not_exists("pose_hrnet_w48_384x288.pth", CHECKPOINT_DIR)
        download_file_if_not_exists("yolov3.weights", CHECKPOINT_DIR)
        keypoints, scores = gen_video_kpts(video_path, det_dim=416, num_person=1, gen_output=True, device=device)
        keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
        # Add conf score to the last dim
        keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    elif yolo_version == "11":
        estimator = YOLOPoseEstimator("yolo11x-pose.pt", CHECKPOINT_DIR, device)
        # Detect person orientation, overwrite video_path with properly oriented video if needed
        video_path = rotate_video_until_upright(video_path)
        keypoints = estimator.get_keypoints_from_video(video_path)

    assert keypoints.ndim == 4, "Keypoints should have 4 dimensions for (num_ppl, num_frames, 17, 3). Received shape: {}".format(keypoints.shape)
    np.save(output_file, keypoints)
    if DEBUG: logger.info(f"2D keypoints (COCO format) saved to {output_file}, with shape {keypoints.shape}")

    return output_file


def load_npy_file(npy_filepath: str) -> np.ndarray:
    """Load keypoints from a .npy file. Use this to load professional 3D keypoints.
    
    Args:
        npy_filepath (str): Path to the professional keypoints .npy file.
    
    Returns:
        np.ndarray: Loaded professional keypoints.
    """
    keypoints_array: np.ndarray = np.load(npy_filepath)
    keypoints_array = keypoints_array.astype(np.float32)
    if DEBUG: logger.info(f"Loaded professional keypoints from {npy_filepath} with shape {keypoints_array.shape} and dype: {keypoints_array.dtype}")
    return keypoints_array

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Refactor the get_pose3D to separate the inference and visualization logic
# GOAL --> code should create the full 3d keypoints array first, then load it later while creating the visualizations
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def get_pose3D_no_vis(
    user_2d_kpts_filepath: str, output_keypoints_path: str, video_path: str, device: str, model_size: str='xs', yaml_path: str=""
):
    """Run 3D pose inference from 2D keypoints using a sliding window (convolutional) approach.
    
    Args:
        user_2d_kpts_filepath (str): Path to the user's 2D keypoints .npy file.
        output_keypoints_path (str): Path to save the output 3D keypoints .npy file.
        video_path (str): Path to the input video file.
        device (str): Device to run the model on ('cpu', 'cuda', or 'mps').
        model_size (str): Size of the MotionAGFormer model ('xs', 's', 'm', 'l').
        yaml_path (str): Path to the YAML configuration file for the model.

    Returns:
        str: Path to the saved 3D keypoints .npy file.
    """
    if yaml_path:
        args = get_config(yaml_path)
        args = {k: v for k, v in args.items() if k in [
            'n_layers', 'dim_in', 'dim_feat', 'dim_rep', 'dim_out',
            'mlp_ratio', 'act_layer',
            'attn_drop', 'drop', 'drop_path',
            'use_layer_scale', 'layer_scale_init_value', 'use_adaptive_fusion',
            'num_heads', 'qkv_bias', 'qkv_scale',
            'hierarchical',
            'use_temporal_similarity', 'neighbour_num', 'temporal_connection_len',
            'use_tcn', 'graph_only',
            'n_frames'
        ]}
        args['act_layer'] = nn.GELU
    else:
        raise ValueError("You must provide a YAML configuration file for the model via the 'yaml_path' argument.")

    if DEBUG: logger.info("\n[INFO] Using MotionAGFormer with the following configuration:")
    if DEBUG: pprint(args)

    # Load model and set to eval() mode to disable training-specific pytorch features
    model = nn.DataParallel(MotionAGFormer(**args)).to(device)
    model_filename = f"motionagformer-{model_size}-h36m*.pth*"
    model_path = download_file_if_not_exists(model_filename, CHECKPOINT_DIR)
    pre_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(pre_dict['model'], strict=True)
    model.eval()
    if DEBUG: logger.info(f"MotionAGFormer model (version: {model_size}) sent to device: {device}, model object type: {type(model)}")

    # Load 2D keypoints
    keypoints = np.load(user_2d_kpts_filepath, allow_pickle=True)  # shape: (1, n_frames, 17, 3)
    n_frames = keypoints.shape[1]
    window_size = args['n_frames']
    half_window = window_size // 2

    # Get video frame size
    cap = cv2.VideoCapture(video_path)
    img_size_h_w = get_frame_size(cap)
    cap.release()

    user_output_3d_keypoints = np.empty((n_frames, 17, 3))

    # Pad keypoints at start/end for edge cases
    pad_left = np.repeat(keypoints[:, 0:1, :, :], half_window, axis=1)
    pad_right = np.repeat(keypoints[:, -1:, :, :], half_window, axis=1)
    padded_keypoints = np.concatenate([pad_left, keypoints, pad_right], axis=1)  # shape: (1, n_frames + 2*half_window, 17, 3)

    for i in tqdm(range(n_frames), desc="MotionAGFormer sliding window", unit="frame"):
        window = padded_keypoints[:, i:i+window_size, :, :]  # shape: (1, window_size, 17, 3)
        input_2D = normalize_screen_coordinates(window, w=img_size_h_w[1], h=img_size_h_w[0])
        input_2D_aug = flip_data(input_2D)
        input_2D = torch.from_numpy(input_2D.astype('float32')).to(device)
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).to(device)

        output_3D_non_flip = model(input_2D)
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        # Only keep the center frame's prediction
        center_idx = window_size // 2
        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, center_idx].cpu().detach().numpy()
        user_output_3d_keypoints[i] = standardize_3d_keypoints(post_out)

    np.save(output_keypoints_path, user_output_3d_keypoints)

    if DEBUG: logger.info(f"3D keypoints saved to {output_keypoints_path}, with shape {user_output_3d_keypoints.shape}")

    return output_keypoints_path



def create_3d_pose_images_from_array(
    user_3d_keypoints_filepath: str,
    output_dir: str,
    pro_keypoints_filepath: str = None,
    is_lefty: bool = False
) -> None:
    """
    Create 3D pose frame visualizations for each frame in the given 3D keypoints array.

    Args:
        user_3d_keypoints (str): Path to a numpy array of shape (N, 17, 3) containing the 3D keypoints for 17 body points and N frames of the user's input video.
        output_dir (str): Output directory to save the images.
        pro_keypoints_filepath (str, optional): Path to the professional keypoints file (for debug/logging).
        is_lefty (bool): If True, flip the professional keypoints horizontally.
    """
    angle_adjustment = 0.0
    # USE_BODY_PART = "feet"
    # USE_BODY_PART = "shoulders"
    USE_BODY_PART = "hips"
    

    # Load professional keypoints and prepare for alignment
    user_keypoints_npy = load_npy_file(user_3d_keypoints_filepath)
    pro_keypoints_npy = load_npy_file(pro_keypoints_filepath)

    # Flip pro keypoints if is_lefty is True
    if is_lefty:
        pro_keypoints_npy = flip_data(pro_keypoints_npy)

    # Pro keypoints should already be standardized, but do it again here just in case
    pro_keypoints_npy = np.array([standardize_3d_keypoints(frame, apply_rotation=False) for frame in pro_keypoints_npy])

    if DEBUG: logger.info(f"User keypoints shape: {user_keypoints_npy.shape}")
    if DEBUG: logger.info(f"Professional keypoints shape: {pro_keypoints_npy.shape}")
    if DEBUG: logger.info(f"user_keypoints_npy dtype: {user_keypoints_npy.dtype}")
    if DEBUG: logger.info(f"pro_keypoints_npy dtype: {pro_keypoints_npy.dtype}")

    # ------------------ Find motion start for both user and pro, then crop ------------------
    user_start, user_end = get_start_end_info(user_keypoints_npy, is_lefty=is_lefty, output_dir = pjoin(output_dir, "user"))
    pro_start, pro_end = get_start_end_info(pro_keypoints_npy, is_lefty=is_lefty, output_dir = pjoin(output_dir, "pro"))

    if DEBUG:
        logger.info(f"User motion starts at frame: {user_start}, ends at frame: {user_end}")
        logger.info(f"Pro motion starts at frame: {pro_start}, ends at frame: {pro_end}")

    user_keypoints_npy = user_keypoints_npy[user_start : user_end]
    pro_keypoints_npy = pro_keypoints_npy[pro_start : pro_end]

    # Note: output_dir points to the 'pose' directory, but pose2D is at the job root level
    job_output_dir = os.path.dirname(output_dir)  # Go up one level from 'pose' to job output root
    pose2d_dir = pjoin(job_output_dir, 'pose2D')
    remove_images_before_after_motion(pose2d_dir, user_start, user_end)

    # Set video_length to the minimum of number of frames between user and pro keypoints files
    num_frames = min(user_keypoints_npy.shape[0], pro_keypoints_npy.shape[0])

    # Resample the longer sequence to match the shorter one
    user_keypoints_npy = resample_pose_sequence(user_keypoints_npy, num_frames)
    pro_keypoints_npy = resample_pose_sequence(pro_keypoints_npy, num_frames)
    if DEBUG: logger.info(f"\nUser keypoints shape after crop/resample: {user_keypoints_npy.shape}")
    if DEBUG: logger.info(f"Professional keypoints shape after crop/resample: {pro_keypoints_npy.shape}")
    # ------------------------------------------------------------------------------------------------

    assert (user_keypoints_npy.shape == pro_keypoints_npy.shape), \
        f"User and professional keypoints must have the same shape after cropping & resampling. Got user: {user_keypoints_npy.shape}, pro: {pro_keypoints_npy.shape}"
    num_frames = user_keypoints_npy.shape[0]

    # Save a copy of the professional keypoints for reference
    job_output_dir = os.path.dirname(output_dir)  # Go up one level to job output root
    raw_keypoints_dir = pjoin(job_output_dir, OUTPUT_FOLDER_RAW_KEYPOINTS)
    output_pro_3D_npy_path = pjoin(raw_keypoints_dir, KEYPOINTS_FILE_3D_PRO)
    np.save(output_pro_3D_npy_path, pro_keypoints_npy)
    if DEBUG: logger.info(f"Professional 3D keypoints saved to {output_pro_3D_npy_path}, with shape {pro_keypoints_npy.shape}")

    for frame_id in tqdm(range(num_frames), desc="Creating 3D pose images", unit="frame"):
        # Create a new figure for this frame
        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05)
        ax = plt.subplot(gs[0], projection='3d')

        user_keypoints_this_frame = user_keypoints_npy[frame_id]

        if pro_keypoints_npy is None:
            show3Dpose(user_keypoints_this_frame, ax)
        else:
            pro_keypoints_this_frame = pro_keypoints_npy[frame_id]
            if frame_id == 0:
                # On the first frame, find the difference between the user and pro stance angle
                user_angle = get_stance_angle(user_keypoints_this_frame, USE_BODY_PART)
                pro_angle = get_stance_angle(pro_keypoints_this_frame, USE_BODY_PART)
                angle_adjustment = user_angle - pro_angle
                if DEBUG:
                    logger.info(f"user_3d_keypoints shape: {user_keypoints_this_frame.shape}")
                    logger.info(f"pro_keypoints_std shape: {pro_keypoints_this_frame.shape}")
                    logger.info(f"Angle between {USE_BODY_PART} in first frame of USER VIDEO: {user_angle:.2f} degrees")
                    logger.info(f"Angle between {USE_BODY_PART} in first frame of PROFESSIONAL VIDEO: {pro_angle:.2f} degrees")
                    logger.info(f"Angle adjustment: {int(angle_adjustment)}")
            # Create the pose overlay image with the user and pro keypoints
            create_pose_overlay_image(
                data1 = user_keypoints_this_frame, 
                data2 = pro_keypoints_this_frame, 
                ax = ax, 
                angle_adjustment = angle_adjustment,  
                use_body_part = USE_BODY_PART,
                show_hip_reference_line = False
            )

        # Save this 3D pose image
        output_path_3D_this_frame = pjoin(output_dir, f"{frame_id:04d}_3D.png")
        plt.savefig(output_path_3D_this_frame, dpi=200, format='png', bbox_inches='tight')
        plt.close(fig)


def remove_images_before_after_motion(pose_img_dir, delete_before_idx = 0, delete_after_idx = None):
    """ Remove pose2D images before and after the specified index in the output directory.
    
    Args:
        dir_type (str): Type of directory to remove images from ('2D' or '3D').
        delete_before_idx (int): Index before which images should be deleted.
        delete_after_idx (int, optional): Index after which images should be deleted. If None, no images will be removed after the specified index.
    """
    if os.path.exists(pose_img_dir):
        print(f"Removing pose2D images before index {delete_before_idx} in directory: {pose_img_dir}")
        pose2d_imgs = sorted([f for f in os.listdir(pose_img_dir) if f.endswith('.png')])

        for i, fname in enumerate(pose2d_imgs):
            img_is_before_motion_start = (i < delete_before_idx) 
            img_is_after_motion_ends = (i > delete_after_idx) and (delete_after_idx is not None) 
            if img_is_before_motion_start or img_is_after_motion_ends:
                os.remove(pjoin(pose_img_dir, fname))
                if DEBUG: logger.info(f"Removed pose2D frame: {fname}")
    else:
        logger.error(f"Directory {pose_img_dir} does not exist. No images to remove.")


def find_motion_start(keypoints: np.ndarray, is_lefty: bool = True, min_pct_change: float = 3) -> int:
    """Find the first frame where the distance between the front foot and sacrum (index 0)
    changes by more than min_pct_change percent relative to the distance in the first frame, using only the z direction.

    Args:
        keypoints (np.ndarray): Shape (n_frames, 17, 3)
        is_lefty (bool): If True, use left foot (index 6) as front foot; else right foot (index 3).
        min_pct_change (float): Minimum relative decrease (as percent) to count as movement.

    Returns:
        int: Index of the first frame where movement is detected.
    """
    # Right Ankle is 3, Left Ankle is 6
    front_foot_idx = 3 if is_lefty else 6 
    sacrum_idx = 0

    if keypoints.ndim == 4:
        keypoints = keypoints[0]

    # Compute initial z distance in the first frame
    initial_dist = abs(keypoints[0, front_foot_idx, 2] - keypoints[0, sacrum_idx, 2])
    if initial_dist == 0:
        return 0  # Avoid division by zero

    for i in range(0, len(keypoints)):
        front_foot_z = keypoints[i, front_foot_idx, 2]
        sacrum_z = keypoints[i, sacrum_idx, 2]

        curr_dist = abs(front_foot_z - sacrum_z)
        pct_change = ((initial_dist - curr_dist) / initial_dist) * 100
        # if DEBUG: logger.info(f"Frame {i}: initial_dist = {initial_dist:.3f}, curr_dist = {curr_dist:.2f}, rel_decrease = {pct_change:.4f}")
        if pct_change > min_pct_change:
            return i
    return 0  # fallback: no movement detected



def get_start_end_info(arr, is_lefty: bool = False, output_dir: str = ".'") -> tuple:
    """Determine the start and end points of the motion based on the ankle positions in the 3D keypoints array.

    Args:
        arr (np.ndarray): Input numpy array, assumed to be of shape (n_frames, 17, 3) for 3D keypoints.

    Returns:
        tuple: A tuple containing (min_value, max_value, average_value).
    """
    assert arr.ndim == 3, "Input array must be 3D with shape (n_frames, 17, 3). Received shape: {}".format(arr.shape)
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    front_ankle_arr = []
    back_ankle_arr = []
    higher_ankle_arr = []
    max_front = 0 # Tracks the max z height of the front ankly
    max_front_index = 100000
    low_point = 0
    starting_point = 0


    # Determine which ankle is front/back based on is_lefty
    if is_lefty:
        front_ankle_name = "Right Ankle"
        back_ankle_name = "Left Ankle"
    else:
        front_ankle_name = "Left Ankle"
        back_ankle_name = "Right Ankle"

    for i in range(len(arr)):
        joints = get_frame_info(arr[i])
        front_ankle_z = joints[front_ankle_name][2]
        back_ankle_z = joints[back_ankle_name][2]

        if front_ankle_z >= max_front:
            max_front = front_ankle_z
            max_front_index = i
            if low_point < i:
                starting_point = low_point

        if front_ankle_z < 0.005:
            low_point = i

        front_ankle_arr.append(front_ankle_z)
        back_ankle_arr.append(back_ankle_z)

    # smooth the arrays
    front_ankle_arr = np.array(front_ankle_arr)
    back_ankle_arr = np.array(back_ankle_arr)
    front_ankle_arr = np.convolve(front_ankle_arr, np.ones(10)/10, mode='valid')
    back_ankle_arr = np.convolve(back_ankle_arr, np.ones(10)/10, mode='valid')

    for i in range(len(front_ankle_arr)):
        if front_ankle_arr[i] > back_ankle_arr[i]:
            higher_ankle_arr.append(1)
        else:
            higher_ankle_arr.append(0)

    # Switch point is the index where the back ankle becomes higher than the front ankle after the max_front_index
    switch_point = 0
    for i in range(max_front_index, len(higher_ankle_arr)):
        if higher_ankle_arr[i] == 0:
            switch_point = i
            break

    logger.info(f"Switch point found: {switch_point}")

    # Find the first point after switch_point where back ankle is below 0.005. If that never happens, end_point is the last frame. 
    end_point = len(arr) - 1  # Default to the last frame
    for i in range(switch_point+5, len(back_ankle_arr)):
        if back_ankle_arr[i] < 0.01:
            end_point = i
            break

    # plot the joints
    plt.plot(front_ankle_arr, label = front_ankle_name + " (Front Ankle)")
    plt.plot(back_ankle_arr, label = back_ankle_name + " (Back Ankle)")
    plt.axvline(x=starting_point, color='r', linestyle='--', label='Starting Point')
    plt.axvline(x=end_point, color='g', linestyle='--', label='End Point')
    plt.legend()
    plt.show()
    save_path  = pjoin(output_dir, "ankle_plot.png")
    logger.info(f" ---> Will save ankle plot to {save_path}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return (starting_point, end_point)


# def get_frame_info(frame: np.ndarray) -> dict:
#     """Get the joint names and their corresponding coordinates from a single frame of 3D key points.

#     Args:
#         frame (np.ndarray): A single frame of 3D key points, expected shape (
#             17, 3), where each row corresponds to a joint and each column corresponds to x, y, z coordinates.

#     Returns:
#         dict: A dictionary mapping joint names to their coordinates. Like {'Hip': [x, y, z], 'Right Hip': [x, y, z], ...}
#     """
#     assert frame.shape == (17, 3), f"Expected frame shape (17, 3), got {frame.shape}. Ensure the input is a single frame of 3D key points."
#     joint_names = [
#         "Hip", "Right Hip", "Right Knee", "Right Ankle",
#         "Left Hip", "Left Knee", "Left Ankle", "Spine",
#         "Thorax", "Neck", "Head", "Left Shoulder",
#         "Left Elbow", "Left Wrist", "Right Shoulder",
#         "Right Elbow", "Right Wrist"
#         ]
#     joints = {joint_names[i]: frame[i] for i in range(len(frame))}
#     # print(f"\n[DEBUG] get_frame_info() - Extracted joints from frame: {joints.keys()}")
#     # pprint(joints)
#     return joints




# -----------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------  END OF BIG REFACTOR  ---------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------


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


def create_2D_images(cap: cv2.VideoCapture, keypoints: np.ndarray, output_dir_2D: str, is_lefty: bool) -> str:
    # output_dir_2D = pjoin(output_dir, 'pose2D')
    # os.makedirs(output_dir_2D, exist_ok=True)
    """Create 2D pose images from keypoints and save them to the specified directory.
    Returns:
        str: Path to the output directory containing the 2D pose images.    
    """
    n_frames = keypoints.shape[1]  
    logger.info(f"Total number of frames in the video: {n_frames} ({keypoints.shape = }")
    logger.info('\n\nGenerating 2D pose images...')
    for i in tqdm(range(n_frames)):
        is_valid, img = cap.read()
        if not is_valid:
            continue
        keypoints_2D_this_frame = keypoints[0][i]
        image_w_keypoints = show2Dpose(keypoints_2D_this_frame, copy.deepcopy(img), is_lefty=is_lefty)
        output_path_img_2D = pjoin(output_dir_2D, f"{i:04d}_2D.png")
        cv2.imwrite(output_path_img_2D, image_w_keypoints)

    return output_dir_2D



def get_stance_angle(data: np.ndarray, use_body_part: str = "feet") -> float:
    """
    Get the angle between the left and right ankles, hips, or shoulders from the 3D pose data.

    Args:
        data (np.ndarray): 3D pose data for a single frame, expected shape (17, 3), where the 2nd dimension contains x, y, z coordinates.
        use_part (str): Which body part to use for angle calculation: "feet", "hips", or "shoulders".

    Returns:
        float: Angle in degrees between the specified keypoints.
    """
    if use_body_part == "hips":
        left = data[1][:2]
        right = data[4][:2]
    elif use_body_part == "shoulders":
        left = data[14][:2]
        right = data[11][:2]
    else:  # Default to "feet"
        left = data[6][:2]
        right = data[3][:2]

    # logger.info(f"\n[DEBUG] get_stance_angle() - left: {left}, right: {right}, use_body_part: {use_body_part}\n")
    assert len(left) == 2 and len(right) == 2, f"Expected left and right to have length 2, got {len(left)} and {len(right)}"

    return find_angle(right, left)


def create_pose_overlay_image(
    data1: np.ndarray, data2: np.ndarray, ax: matplotlib.axis,
    angle_adjustment: float = 0.0, use_body_part: str = "hips",
    show_hip_reference_line: bool = False
) -> str:
    """Create a single image with 3D pose keypoints from data1 and data2 overlaid on the same axis.
    Both data1 and data2 are standardized before plotting.

    Args:
        data1 (np.ndarray): 3D pose data for the user, expected shape (17, 3).
        data2 (np.ndarray): 3D pose data for the professional, expected shape (17, 3).
        ax: Matplotlib 3D axis to plot on.
        angle_adjustment (float): Angle adjustment in degrees to align the pro pose with the user pose.
        use_body_part (str): Which body part to use for angle calculation: "feet", "hips", or "shoulders".
        show_hip_reference_line (bool): Whether to draw reference lines for the hip angles. Default is False.
    
    Returns:
        str: Path to the saved overlay image.
        
    """
    # Rotate the pro pose to align with the user pose based on the angle adjustment from the first frame
    # logger.info(f"[ DEBUG create_pose_overlay_image() ], {data1.shape =}, {data2.shape =}, {angle_adjustment = }, {use_body_part = }, {data1.dtype = }, {data2.dtype = }")
    data2 = rotate_along_z(data2, angle_adjustment)

    # Recenter both poses on their left ankles
    # 6 is left ankle, 0 is sacrum (middle of hips)
    keypoint_in_center = 0
    data1 = recenter_on_joint(data1, keypoint_in_center) 
    data2 = recenter_on_joint(data2, keypoint_in_center)
    
    if show_hip_reference_line:
        d1_angle = get_stance_angle(data1, use_body_part)
        d2_angle = get_stance_angle(data2, use_body_part)
        draw_reference_angle_line(ax, d1_angle, color='blue', linewidth=2)
        draw_reference_angle_line(ax, d2_angle, color='green', linewidth=2)
        # if DEBUG: logger.info("\nuser stance angle =", int(d1_angle))
        # if DEBUG: logger.info("pro  stance angle =", int(d2_angle))

    # Visualize the aligned poses
    # Plot the user on top of the pro pose
    show3Dpose(data2, ax, color='blk')
    show3Dpose(data1, ax, color='R')
    return None


def draw_reference_angle_line(ax, angle_degrees, color='k', linewidth=2) -> None:
    """
    Draws a line in 3D space at z=0 from one edge of a box (x,y in [-1,1]) through (0,0,0)
    at the specified angle (degrees, counterclockwise from x-axis).

    Args:
        ax: Matplotlib 3D axis.
        angle_degrees (float): Angle in degrees (0 = along +x axis, 90 = along +y axis).
        color (str): Line color.
        linewidth (int): Line width.
    """
    theta = np.deg2rad(angle_degrees)
    # Direction vector
    dx = np.cos(theta)
    dy = np.sin(theta)

    # Find intersection with box edge (x or y = ±1)
    # Parametric: (x, y) = t*(dx, dy), find t so x or y hits ±1
    t_x = np.inf if dx == 0 else 1.0 / abs(dx)
    t_y = np.inf if dy == 0 else 1.0 / abs(dy)
    t = min(t_x, t_y)

    # Endpoints: from -t to +t (so line goes edge-to-edge)
    x0, y0 = -dx * t, -dy * t
    x1, y1 = dx * t, dy * t

    # Clip to box [-1,1]
    def clip(val):
        return max(-1, min(1, val))

    # If the line is not axis-aligned, the above will work.
    # But for completeness, clip both endpoints to the box.
    x0, y0 = clip(x0), clip(y0)
    x1, y1 = clip(x1), clip(y1)
    ax.plot([x0, x1], [y0, y1], [0, 0], color=color, linewidth=linewidth, alpha=0.5)



def standardize_3d_keypoints(keypoints: np.ndarray, apply_rotation: bool = True) -> np.ndarray:
    """
    Standardizes 3D keypoints by transforming them from camera coordinates to world coordinates,
    shifting them so that the minimum z value is at zero (ground level), and normalizing 
    them so that the largest value is 1.

    Args:
        keypoints (np.ndarray): 3D keypoints of shape (N, 17, 3) where N is the number of frames.
        apply_rotation (bool): Whether to apply a rotation transformation to the keypoints. 
            Should be False when standardizing the PRO PLAYER keypoints.
        
    Returns:
        np.ndarray: Standardized 3D keypoints of shape (N, 17, 3).
    """
    if apply_rotation:
        # Define a quaternion rotation (from model or calibration)
        rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    else:
        # Professional keypoints are already in world coordinates, no rotation needed
        rot = [0.0, 0.0, 0.0, 0.0]

    # Transform the keypoints from camera coordinates to world coordinates using the rotation
    rot = np.array(rot, dtype='float32')
    keypoints = camera_to_world(keypoints, R=rot, t=0)
    
    # Shift all keypoints so that the minimum z value is at zero (ground level)
    keypoints[:, 2] -= np.min(keypoints[:, 2])
    
    # Normalize keypoints to be between 0 and 1 (scaling for visualization/consistency)
    max_value = np.max(keypoints)
    keypoints /= max_value
    return keypoints


def generate_output_combined_frames(output_dir_2D: str, output_dir_3D: str, output_dir_combined: str) -> None:
    """Generate a demo video showing 2D input and 3D reconstruction side by side."""
    logger.info('\n\nGenerating demo video frames...')
    
    # Efficient batch processing for demo video generation
    # Accept both *_2D.png and *.png for 2D images (for compatibility)
    image_2d_paths = sorted(glob.glob(pjoin(output_dir_2D, '*_2D.png')))
    if not image_2d_paths:
        image_2d_paths = sorted(glob.glob(pjoin(output_dir_2D, '*.png')))
    image_3d_paths = sorted(glob.glob(pjoin(output_dir_3D, '*.png')))

    n_frames = min(len(image_2d_paths), len(image_3d_paths))
    if n_frames == 0:
        logger.info("No frames found for demo video generation.")
        return

    logger.info('\n\nGenerating demo...')
    # output_dir_pose = pjoin(output_dir, 'pose')
    # os.makedirs(output_dir_pose, exist_ok=True)

    FONT_SIZE = 12
    for i in tqdm(range(n_frames), desc="Generating output video frames", unit="frame"):
        img2d = plt.imread(image_2d_paths[i])
        img3d = plt.imread(image_3d_paths[i])
        # Only crop 2D if it is wider than tall (avoid empty images)
        if img2d.shape[1] > img2d.shape[0]:
            edge2d = (img2d.shape[1] - img2d.shape[0]) // 2
            img2d = img2d[:, edge2d:img2d.shape[1] - edge2d]
        # Defensive: if after crop, shape is invalid, skip crop
        if img2d.shape[0] == 0 or img2d.shape[1] == 0:
            img2d = plt.imread(image_2d_paths[i])
        # Crop 3D as before
        edge3d = 130
        if img3d.shape[0] > 2 * edge3d and img3d.shape[1] > 2 * edge3d:
            img3d = img3d[edge3d:img3d.shape[0] - edge3d, edge3d:img3d.shape[1] - edge3d]

        fig, axs = plt.subplots(1, 2, figsize=(15.0, 5.4))
        # Remove axes for both
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
        axs[0].imshow(img2d)
        axs[0].set_title("Input", fontsize=FONT_SIZE)
        axs[1].imshow(img3d)
        axs[1].set_title("Reconstruction", fontsize=FONT_SIZE)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        output_path_pose_thisimg = pjoin(output_dir_combined, f"{i:04d}_pose.png")
        fig.savefig(output_path_pose_thisimg, dpi=200, bbox_inches='tight')
        plt.close(fig)



def img2video(video_path: str, input_frames_dir: str) -> str:
    """Converts a sequence of pose images into a video.

    Args:
        video_path (str): Path to the original input video (used for FPS and naming).
        input_frames_dir (str): Directory containing the 'pose' subdirectory with PNG frames.

    Raises:
        FileNotFoundError: If the 'pose' directory does not exist.
        FileNotFoundError: If no PNG frames are found in the 'pose' directory.
        ValueError: If the first pose image cannot be read.

    Returns:
        str: Path to the generated output video file (a .mp4 file).
    """
    import logging
    logger = logging.getLogger(__name__)
    video_name = video_path.split('/')[-1].split('.')[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps):
        fps = 25  # fallback default
    cap.release()

    pose_filenames = sorted(glob.glob(pjoin(input_frames_dir, '*.png')))
    if not pose_filenames:
        logger.error(f"No pose PNG frames found in {input_frames_dir}")
        raise FileNotFoundError(f"No pose PNG frames found in {input_frames_dir}")
    
    img = cv2.imread(pose_filenames[0])
    if img is None:
        logger.error(f"Failed to read first pose image: {pose_filenames[0]}")
        raise ValueError(f"Failed to read first pose image: {pose_filenames[0]}")
    else:
        size = (img.shape[1], img.shape[0])

    output_video_name = video_name.replace("input", "output")
    output_path = pjoin(input_frames_dir, output_video_name + '.mp4')
    logger.info(f"Writing output video to: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWrite = cv2.VideoWriter(output_path, fourcc, fps, size)

    for name in pose_filenames:
        img = cv2.imread(name)
        if img is None:
            logger.warning(f"Failed to read pose image: {name}, skipping.")
            continue
        # Ensure the frame size matches the video size
        if (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size)
        videoWrite.write(img)

    videoWrite.release()
    logger.info(f"Video writing complete: {output_path}")
    return output_path


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def resample(n_frames, target_frames):
    even = np.linspace(0, n_frames, num=target_frames, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result



def turn_into_clips(keypoints, target_frames):
    clips = []
    n_frames = keypoints.shape[1]
    downsample = None
    if n_frames <= target_frames:
        new_indices = resample(n_frames, target_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, target_frames):
            keypoints_clip = keypoints[:, start_idx:start_idx + target_frames, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != target_frames:
                new_indices = resample(clip_length, target_frames)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
        if downsample is None:
            # If all clips are full length, set downsample to default (all indices)
            downsample = np.arange(target_frames)
    return clips, downsample

def turn_into_h36m(keypoints):
    new_keypoints = np.zeros_like(keypoints)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 11, :]
    new_keypoints[..., 2, :] = keypoints[..., 13, :]
    new_keypoints[..., 3, :] = keypoints[..., 15, :]
    new_keypoints[..., 4, :] = keypoints[..., 12, :]
    new_keypoints[..., 5, :] = keypoints[..., 14, :]
    new_keypoints[..., 6, :] = keypoints[..., 16, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    new_keypoints[..., 11, :] = keypoints[..., 6, :]
    new_keypoints[..., 12, :] = keypoints[..., 8, :]
    new_keypoints[..., 13, :] = keypoints[..., 10, :]
    new_keypoints[..., 14, :] = keypoints[..., 5, :]
    new_keypoints[..., 15, :] = keypoints[..., 7, :]
    new_keypoints[..., 16, :] = keypoints[..., 9, :]

    return new_keypoints


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data
    

def rotate_along_z(kpts: np.ndarray, degrees: float) -> np.ndarray:
    """
    Rotates a set of 3D keypoints around the Z-axis.

    Parameters:
        kpts (np.ndarray): Array of shape (17, 3) containing [x, y, z] keypoints.
        degrees (float): Angle in degrees to rotate around the Z-axis.

    Returns:
        np.ndarray: Rotated keypoints of shape (17, 3).
    """
    assert kpts.shape == (17, 3), "Input must be of shape (17, 3)"

    radians = np.deg2rad(degrees)
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)

    # Rotation matrix for Z-axis
    Rz = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,               0,    1]
    ])

    # Rotate each point
    rotated_kpts = kpts @ Rz.T  # matrix multiply (17x3) x (3x3).T = (17x3)

    return rotated_kpts


def recenter_on_joint(pose: np.ndarray, keypoint_idx: int = 6, target_xyz=(0.0, 0.0, 0.0)):
    """
    Recenter a pose so that the specified ankle joint moves to the given target coordinates.

    Parameters:
    - pose (np.ndarray): Shape (17, 3) containing joint coordinates
    - keypoint_idx (int): Index of the ankle joint (6 for left ankle, 3 for right ankle)
    - target_xyz (tuple): The target (x, y, z) coordinates to move the ankle to

    Returns:
    - np.ndarray: Recentered pose
    """
    keypoint_xyz = pose[keypoint_idx:keypoint_idx+1, :]  # shape (1, 3)
    target = np.array(target_xyz)  # shape (3,)
    return pose - keypoint_xyz + target


def find_angle(coord1, coord2):
    """Finds the angle in degrees between the line segment connecting two 2D coordinates
    and the positive x-axis.

    Args:
        coord1 (tuple): A tuple representing the first (x, y) coordinate.
        coord2 (tuple): A tuple representing the second (x, y) coordinate.

    Returns:
        float: The angle in degrees. Returns None if the points are the same.
    """
    # logger.info(f"[ DEBUG find_angle() ], {coord1 = }")
    # logger.info(f"[ DEBUG find_angle() ], {coord2 = }")
    x1, y1 = coord1
    x2, y2 = coord2

    # if DEBUG: logger.info(f"[ DEBUG find_angle() ], {x1 = }, {y1 = }, {x2 = }, {y2 = }")
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return None  # Points are the same, angle is undefined

    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle

if __name__ == "__main__":
    # Choose model size and config file to use:
    # 'xs', 's', 'b', 'l'

    MODEL_SIZE = 'xs'
    MODEL_CONFIG_PATH = "./configs/h36m/MotionAGFormer-xsmall.yaml"
    # MODEL_SIZE = 's'
    # MODEL_CONFIG_PATH = "./configs/h36m/MotionAGFormer-small.yaml"
    # MODEL_SIZE = 'b'
    # MODEL_CONFIG_PATH = "./configs/h36m/MotionAGFormer-base.yaml"
    # MODEL_SIZE = 'l'
    # MODEL_CONFIG_PATH = "./configs/h36m/MotionAGFormer-large.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = pjoin('.', 'demo', 'video', args.video)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = pjoin('.', 'demo', 'output', video_name)

    device = get_pytorch_device()
    logger.info(f"Using device: {device}")
    
    # get_pose2D(video_path, output_dir, device)
    # get_pose3D(video_path, output_dir, device, MODEL_SIZE, MODEL_CONFIG_PATH)
    # img2video(video_path, output_dir)
    # logger.info('Generating demo successful!')
