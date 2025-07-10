import argparse
import copy
import glob
import os

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
from src.utils import download_file_if_not_exists, normalize_screen_coordinates, camera_to_world, get_config
from src.model.MotionAGFormer import MotionAGFormer
from src.visualizations import resample_pose_sequence

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ---------------------- CONSTANTS ----------------------
BACKEND_API_DIR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        'R':   ((0.6, 0.0, 0.0), (1.0, 0.0, 0.0)),     # Dark Red, Bright Red
        'G':   ((0.0, 0.6, 0.0), (0.0, 1.0, 0.0)),     # Dark Green, Bright Green
        'blk': ((0.4, 0.4, 0.4), (0.2, 0.2, 0.2)),     # Light gray, Dark gray
    }

    lcolor, rcolor = format_options.get(color, ((0,0,1),(1,0,0)))  # Default to Blue, Red

    if use_0_255_range:
        # Adjust for OpenCV's default color format of range 0-255 and BGR order
        lcolor = lcolor[::-1]
        rcolor = rcolor[::-1]
        lcolor = tuple(int(255 * c) for c in lcolor)
        rcolor = tuple(int(255 * c) for c in rcolor)

    return (lcolor, rcolor)



def show2Dpose(kps: np.ndarray, img: np.ndarray, color = 'R') -> np.ndarray:
    """
    Draws a 2D human pose skeleton on the given image using the provided keypoints.

    Args:
        kps (np.ndarray): An array of shape (N, 2) containing the 2D coordinates of keypoints.
        img (np.ndarray): The image (as a NumPy array) on which to draw the skeleton.

    Returns:
        np.ndarray: The image with the 2D pose skeleton drawn on it.
    """
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    # 1 = left arm & leg, 0 = right arm/leg, torso, neck
    LR = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor, rcolor = get_joint_colors(color, use_0_255_range=True)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

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

    show_axis_label = True
    ax.tick_params('x', labelbottom=show_axis_label)
    ax.tick_params('y', labelleft=show_axis_label)
    ax.tick_params('z', labelleft=show_axis_label)

    # Plot small black dots at each keypoint
    vals = np.asarray(vals)
    ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], c='k', s=3, depthshade=False, alpha=0.5)



def get_pose2D(video_path, output_dir, device):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')

    # Check if HRNet checkpoint exists, if not, download from S3
    checkpoint_dir = os.path.join(BACKEND_API_DIR_ROOT, "checkpoint")
    download_file_if_not_exists("pose_hrnet_w48_384x288.pth", checkpoint_dir)
    download_file_if_not_exists("yolov3.weights", checkpoint_dir)

    keypoints, scores = gen_video_kpts(video_path, det_dim=416, num_person=2, gen_output=True, device=device)

    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)

    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_dir = os.path.join(output_dir, OUTPUT_FOLDER_RAW_KEYPOINTS)
    os.makedirs(output_dir, exist_ok=True)

    output_npy = os.path.join(output_dir, KEYPOINTS_FILE_2D)
    np.save(output_npy, keypoints)
    if DEBUG: print(f"2D keypoints saved to {output_npy}, with shape {keypoints.shape}")



def load_npy_file(npy_filepath: str) -> np.ndarray:
    """Load keypoints from a .npy file. Use this to load professional 3D keypoints.
    
    Args:
        npy_filepath (str): Path to the professional keypoints .npy file.
    
    Returns:
        np.ndarray: Loaded professional keypoints.
    """
    keypoints_array = np.load(npy_filepath)
    if DEBUG: print(f"Loaded professional keypoints from {npy_filepath} with shape {keypoints_array.shape}")
    return keypoints_array



@torch.no_grad()
def get_pose3D(
    video_path: str, output_dir: str, device: str, model_size: str='*', yaml_path: str=None,
    pro_keypoints_filepath: str = None
    ): 
    """
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the output will be saved.
        device (str): Device to run the model on ('cpu', 'cuda', or 'mps').
        model_size (str): Size of the model to use ('xs', 's', 'b', 'l').
        yaml_path (str): Path to the YAML configuration file for the model.
        pro_keypoints_filepath (str): Path to the professional keypoints .npy file for alignment.

    Raises:
        ValueError: If the YAML configuration file is not provided.
    """
    if yaml_path is not None:
        args=get_config(yaml_path)
        # Filter args to only include those in the "Model" section
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

    if DEBUG: print("\n[INFO] Using MotionAGFormer with the following configuration:")
    if DEBUG: pprint(args)

    # ---------------------- Reload the model ----------------------
    model = nn.DataParallel(MotionAGFormer(**args)).to(device)
    if DEBUG: print(f"{type(model) = }")

    checkpoint_dir = os.path.join(BACKEND_API_DIR_ROOT, "checkpoint")
    if DEBUG: print(f"[INFO] Checkpoint directory: {checkpoint_dir}")
    model_filename = f"motionagformer-{model_size}-h36m*.pth*"
    model_path = download_file_if_not_exists(model_filename, checkpoint_dir)

    pre_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(pre_dict['model'], strict=True)
    model.eval()

    # -------- load 2D keypoints numpy file, create 2D images in pose2D folder --------
    ## input
    keypoints_path = os.path.join(output_dir, OUTPUT_FOLDER_RAW_KEYPOINTS, KEYPOINTS_FILE_2D)
    keypoints = np.load(keypoints_path, allow_pickle=True)

    clips, downsample = turn_into_clips(keypoints=keypoints, target_frames=args['n_frames'])

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ## 3D
    print('\nGenerating 2D pose image...')
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if img is None:
            continue
        img_size = img.shape
        input_2D = keypoints[0][i]

        image = show2Dpose(input_2D, copy.deepcopy(img))

        output_dir_2D = os.path.join(output_dir, 'pose2D')
        os.makedirs(output_dir_2D, exist_ok=True)
        output_path_2D = os.path.join(output_dir_2D, f"{i:04d}_2D.png")
        cv2.imwrite(output_path_2D, image)


    # ----------- Setup for 3D Pose Generation -----------
    # Load professional keypoints and prepare for alignment
    pro_keypoints_npy = load_npy_file(pro_keypoints_filepath)
    
    # Check if we need to resample the professional keypoints to match our video length
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(pro_keypoints_npy) != video_length:
        print(f"\nResampling professional keypoints from {len(pro_keypoints_npy)} to {video_length} frames")
        pro_keypoints_npy = resample_pose_sequence(pro_keypoints_npy, video_length)
    
    pro_keypoints_std = np.array([standardize_3d_keypoints(frame) for frame in pro_keypoints_npy])
    user_output_3d_keypoints = np.empty_like(pro_keypoints_std)

    # ----------- Run 3D Pose Generation -----------
    print('\nGenerating 3D pose...')
    # for clip_idx, clip in tqdm(enumerate(clips)):
    for clip_idx in tqdm(range(len(clips))):
        clip = clips[clip_idx]

        if DEBUG: print(f"Processing clip {clip_idx + 1}/{len(clips)}...")

        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 
        input_2D_aug = flip_data(input_2D)

        input_2D = torch.from_numpy(input_2D.astype('float32')).to(device)
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).to(device)

        # MotionAGFormer inference on input and flipped input 2d keypoints, then unflip the flipped 2d values
        # Resulting 3D keypoints are the average of both 
        output_3D_non_flip = model(input_2D) 
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if clip_idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()

        if DEBUG: print(f"post_out_all shape: {post_out_all.shape}")
        if DEBUG: print(f"pro_keypoints_std shape: {pro_keypoints_std.shape}")

        # Create 3d pose frames for the output video
        for j, post_out in enumerate(post_out_all):
            frame_id = clip_idx * args['n_frames'] + j

            # Set the output filepath for this frame
            output_dir_3D = os.path.join(output_dir, 'pose3D')
            os.makedirs(output_dir_3D, exist_ok=True)
            output_path_3D = os.path.join(output_dir_3D, f"{frame_id:04d}_3D.png")

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05) 
            ax = plt.subplot(gs[0], projection='3d')
            
            post_out = standardize_3d_keypoints(post_out)
            pro_keypoints_std = standardize_3d_keypoints(pro_keypoints_npy[frame_id], apply_rotation=False)

            if pro_keypoints_filepath is not None:    
                # If we have professional keypoints, we will overlay them on the user's pose
                if (frame_id) == 0: 
                    # On the first frame, find the difference between the user and pro stance angle, determined by the chosen body parts
                    # Apply that same rotation adjustment about the Z axis to the Pro's stance for all other frames so that 
                    # the professional keypoints are aligned with the user's stance from the start
                    
                    USE_BODY_PART = "hips"
                    pro_feet_angle = get_stance_angle(pro_keypoints_std, USE_BODY_PART)
                    user_feet_angle = get_stance_angle(post_out, USE_BODY_PART)
                    angle_adjustment = user_feet_angle - pro_feet_angle
                    if DEBUG: 
                        print("post_out shape:", post_out.shape)
                        print("pro_post_out_std shape:", pro_keypoints_std.shape)
                        print(f"Angle between {USE_BODY_PART} in first frame of USER VIDEO: {user_feet_angle:.2f} degrees")
                        print(f"Angle between {USE_BODY_PART} in first frame of PROFESSIONAL VIDEO: {pro_feet_angle:.2f} degrees")
                        print("Angle adjustment:", int(angle_adjustment))

                create_pose_overlay_image(post_out, pro_keypoints_std, ax, output_path_3D, angle_adjustment, USE_BODY_PART)

                # Save the 3D pose image with the overlay
                # Append post_out as a new layer to user_output_3d_keypoints
                if DEBUG: print(f"Output 3D keypoints shape before append: {user_output_3d_keypoints.shape}")
                user_output_3d_keypoints[frame_id] = post_out
                if DEBUG: print(f"Output 3D keypoints shape after append: {user_output_3d_keypoints.shape}")

                if DEBUG: print(f"Overlay image saved to {output_path_3D}")
            else:
                # If no professional keypoints are provided, just show the user's pose
                show3Dpose(post_out, ax)
            
            # Save the 3d pose image as png
            plt.savefig(output_path_3D, dpi=200, format='png', bbox_inches='tight')
            plt.close(fig)

    # Save the 3D keypoints to a .npy file
    output_3D_npy = os.path.join(output_dir, OUTPUT_FOLDER_RAW_KEYPOINTS, KEYPOINTS_FILE_3D_USER)
    np.save(output_3D_npy, user_output_3d_keypoints)
    if DEBUG: print(f"3D keypoints saved to {output_3D_npy}, with shape {user_output_3d_keypoints.shape}")

    # Save a copy of the professional keypoints for reference
    output_pro_3D_npy = os.path.join(output_dir, OUTPUT_FOLDER_RAW_KEYPOINTS, KEYPOINTS_FILE_3D_PRO)
    np.save(output_pro_3D_npy, pro_keypoints_npy)
    if DEBUG: print(f"Professional 3D keypoints saved to {output_pro_3D_npy}, with shape {pro_keypoints_npy.shape}")

    print('Generating 3D pose successful!')
    generate_demo_video(output_dir_2D, output_dir_3D, output_dir)

    return output_3D_npy


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
    return find_angle(right, left)


def create_pose_overlay_image(
    data1: np.ndarray, data2: np.ndarray, ax: matplotlib.axis, output_path: str, 
    angle_adjustment: float = 0.0, use_body_part: str = "feet",
    show_hip_reference_line: bool = False,
    view_angle: float = 0.0, camera_height: float = 15.0
) -> str:
    """Create a single image with 3D pose keypoints from data1 and data2 overlaid on the same axis.
    Both data1 and data2 are standardized before plotting.

    Args:
        data1 (np.ndarray): 3D pose data for the user, expected shape (17, 3).
        data2 (np.ndarray): 3D pose data for the professional, expected shape (17, 3).
        ax: Matplotlib 3D axis to plot on.
        output_path (str): Path to save the output image.
        angle_adjustment (float): Angle adjustment in degrees to align the pro pose with the user pose.
        use_body_part (str): Which body part to use for angle calculation: "feet", "hips", or "shoulders".
        show_hip_reference_line (bool): Whether to draw reference lines for the hip angles. Default is False.
        view_angle (float): Elevation angle for the camera view.
        camera_height (float): Height of the camera view.
    
    Returns:
        str: The output path where the image is saved.
    """
    # Rotate the pro pose to align with the user pose based on the angle adjustment from the first frame
    data2 = rotate_along_z(data2, angle_adjustment)

    # Recenter both poses on their left ankles
    # 6 is left ankle, 0 is sacrum (middle of hips)
    keypoint_in_center = 0
    data1 = recenter_on_joint(data1, keypoint_in_center) 
    data2 = recenter_on_joint(data2, keypoint_in_center)
    
    d1_angle = get_stance_angle(data1, use_body_part)
    d2_angle = get_stance_angle(data2, use_body_part)
    if DEBUG: print("\nuser stance angle =", int(d1_angle))
    if DEBUG: print("pro  stance angle =", int(d2_angle))

    if show_hip_reference_line:
        draw_reference_angle_line(ax, d1_angle, color='blue', linewidth=2)
        draw_reference_angle_line(ax, d2_angle, color='green', linewidth=2)

    # Visualize the aligned poses
    # Plot the user on top of the pro pose
    show3Dpose(data2, ax, color='blk')
    show3Dpose(data1, ax, color='R')
    return output_path


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


def generate_demo_video(output_dir_2D: str, output_dir_3D: str, output_dir: str) -> None:
    """Generate a demo video showing 2D input and 3D reconstruction side by side."""
    # Efficient batch processing for demo video generation
    image_2d_paths = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_paths = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    n_frames = min(len(image_2d_paths), len(image_3d_paths))
    if n_frames == 0:
        print("No frames found for demo video generation.")
        return

    print('\nGenerating demo...')
    output_dir_pose = os.path.join(output_dir, 'pose')
    os.makedirs(output_dir_pose, exist_ok=True)

    # Preload all images into memory for faster access (if memory allows)
    images_2d = []
    images_3d = []
    for i in range(n_frames):
        img2d = plt.imread(image_2d_paths[i])
        img3d = plt.imread(image_3d_paths[i])
        # Crop 2D
        edge2d = (img2d.shape[1] - img2d.shape[0]) // 2
        img2d = img2d[:, edge2d:img2d.shape[1] - edge2d]
        # Crop 3D
        edge3d = 130
        img3d = img3d[edge3d:img3d.shape[0] - edge3d, edge3d:img3d.shape[1] - edge3d]
        images_2d.append(img2d)
        images_3d.append(img3d)

    # Use Agg canvas directly for speed, avoid repeated plt.figure/close
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    font_size = 12
    for i in tqdm(range(n_frames)):
        fig, axs = plt.subplots(1, 2, figsize=(15.0, 5.4))
        # Remove axes for both
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
        axs[0].imshow(images_2d[i])
        axs[0].set_title("Input", fontsize=font_size)
        axs[1].imshow(images_3d[i])
        axs[1].set_title("Reconstruction", fontsize=font_size)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        output_path_pose = os.path.join(output_dir_pose, f"{i:04d}_pose.png")
        fig.savefig(output_path_pose, dpi=200, bbox_inches='tight')
        plt.close(fig)



def img2video(video_path: str, output_dir: str) -> str:
    """Converts a sequence of pose images into a video.

    Args:
        video_path (str): Path to the original input video (used for FPS and naming).
        output_dir (str): Directory containing the 'pose' subdirectory with PNG frames.

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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    pose_dir = os.path.join(output_dir, 'pose')
    if not os.path.exists(pose_dir):
        logger.error(f"Pose directory does not exist: {pose_dir}")
        raise FileNotFoundError(f"Pose directory does not exist: {pose_dir}")
    
    pose_filenames = sorted(glob.glob(os.path.join(pose_dir, '*.png')))
    if not pose_filenames:
        logger.error(f"No pose PNG frames found in {pose_dir}")
        raise FileNotFoundError(f"No pose PNG frames found in {pose_dir}")
    
    img = cv2.imread(pose_filenames[0])
    if img is None:
        logger.error(f"Failed to read first pose image: {pose_filenames[0]}")
        raise ValueError(f"Failed to read first pose image: {pose_filenames[0]}")
    
    output_video_name = video_name.replace("input", "output")
    output_path = os.path.join(output_dir, output_video_name + '.mp4')
    logger.info(f"Writing output video to: {output_path}")
    size = (img.shape[1], img.shape[0])
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


def get_pytorch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    

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
    x1, y1 = coord1
    x2, y2 = coord2

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

    video_path = os.path.join('.', 'demo', 'video', args.video)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join('.', 'demo', 'output', video_name)

    device = get_pytorch_device()
    print(f"Using device: {device}")
    
    get_pose2D(video_path, output_dir, device)
    get_pose3D(video_path, output_dir, device, MODEL_SIZE, MODEL_CONFIG_PATH)
    img2video(video_path, output_dir)
    print('Generating demo successful!')