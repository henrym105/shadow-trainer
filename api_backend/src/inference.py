import argparse
import cv2
import os 
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
from pprint import pprint

# from utils import get_config
from src.preprocess import h36m_coco_format
from src.hrnet.gen_kpts import gen_video_kpts
from src.utils import get_or_download_checkpoint, normalize_screen_coordinates, camera_to_world, get_config
from src.model.MotionAGFormer import MotionAGFormer

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def show2Dpose(kps: np.ndarray, img: np.ndarray) -> np.ndarray:
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

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
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


def show3Dpose(vals: np.ndarray, ax) -> None:
    """Adds visualization of 3D human pose keypoints on the given matplotlib 3D axis.

    Args:
        vals (np.ndarray): An array of shape (N, 3) representing the 3D coordinates of N joints.
        ax (matplotlib.axes._subplots.Axes3DSubplot): A matplotlib 3D axis object to plot the pose on.

    Returns:
        None
    """
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def get_pose2D(video_path, output_dir, device):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')

    # Check if HRNet checkpoint exists, if not, download from S3
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoint")
    get_or_download_checkpoint("pose_hrnet_w48_384x288.pth", checkpoint_dir)
    get_or_download_checkpoint("yolov3.weights", checkpoint_dir)

    keypoints, scores = gen_video_kpts(video_path, det_dim=416, num_person=2, gen_output=True, device=device)

    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)

    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_dir = os.path.join(output_dir, 'input_2D')
    os.makedirs(output_dir, exist_ok=True)

    output_npz = os.path.join(output_dir, 'keypoints.npz')
    np.savez_compressed(output_npz, reconstruction=keypoints)



@torch.no_grad()
def get_pose3D(video_path: str, output_dir: str, device: str, model_size: str='*', yaml_path: str=None):
    """Generate 3D pose estimation from video input.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the output will be saved.
        device (str): Device to run the model on ('cpu', 'cuda', or 'mps').
        model_size (str): Size of the model to use ('xs', 's', 'b', 'l').
        yaml_path (str): Path to the YAML configuration file for the model.

    Raises:
        ValueError: If the YAML configuration file is not provided.
    """
    # Load model configuration
    args = _load_model_config(yaml_path)
    
    # Setup and load model
    model = _setup_and_load_model(args, device, model_size)

    # Load input keypoints
    keypoints_path = os.path.join(output_dir, 'input_2D', 'keypoints.npz')
    keypoints = np.load(keypoints_path, allow_pickle=True)['reconstruction']
    clips, downsample = turn_into_clips(keypoints, args['n_frames'])

    # Get video properties
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Generate 2D pose images
    img_size, output_dir_2D = _generate_2D_pose_images(video_path, output_dir, keypoints, video_length)
    
    # Generate 3D pose predictions
    user_pose3d_npy = _generate_3D_pose_predictions(model, clips, downsample, img_size, device, args, output_dir)
    
    # Generate combined demo images
    _generate_demo_images(output_dir_2D, output_dir_3D, output_dir)

    return 

def _load_model_config(yaml_path: str) -> dict:
    """Load and filter model configuration from YAML file.
    
    Args:
        yaml_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Filtered model configuration arguments.
        
    Raises:
        ValueError: If yaml_path is None.
    """
    if yaml_path is None:
        raise ValueError("You must provide a YAML configuration file for the model via the 'yaml_path' argument.")
    
    args = get_config(yaml_path)
    # Filter args to only include those in the "Model" section
    filtered_args = {k: v for k, v in args.items() if k in [
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
    filtered_args['act_layer'] = nn.GELU
    
    print("\n[INFO] Using MotionAGFormer with the following configuration:")
    pprint(filtered_args)
    
    return filtered_args


def _setup_and_load_model(args: dict, device: str, model_size: str) -> nn.Module:
    """Setup MotionAGFormer model and load checkpoint weights.
    
    Args:
        args (dict): Model configuration arguments.
        device (str): Device to run the model on.
        model_size (str): Size of the model to use.
        
    Returns:
        nn.Module: Loaded and initialized model.
    """
    model = nn.DataParallel(MotionAGFormer(**args)).to(device)
    print(f"{type(model) = }")

    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoint")
    model_filename = f"motionagformer-{model_size}-h36m*.pth*"
    model_path = get_or_download_checkpoint(model_filename, checkpoint_dir)

    pre_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(pre_dict['model'], strict=True)
    model.eval()
    
    return model


def _generate_2D_pose_images(video_path: str, output_dir: str, keypoints: np.ndarray, video_length: int) -> tuple:
    """Generate 2D pose visualization images from keypoints.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where output will be saved.
        keypoints (np.ndarray): 2D keypoints array.
        video_length (int): Number of frames in the video.
        
    Returns:
        tuple: Image size (height, width, channels) and output directory for 2D poses.
    """
    cap = cv2.VideoCapture(video_path)
    output_dir_2D = os.path.join(output_dir, 'pose2D')
    os.makedirs(output_dir_2D, exist_ok=True)
    
    print('\nGenerating 2D pose image...')
    img_size = None
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if img is None:
            continue
        img_size = img.shape

        input_2D = keypoints[0][i]
        image = show2Dpose(input_2D, copy.deepcopy(img))

        output_path_2D = os.path.join(output_dir_2D, f"{i:04d}_2D.png")
        cv2.imwrite(output_path_2D, image)
    
    cap.release()
    return img_size, output_dir_2D


@torch.no_grad()
def _generate_3D_pose_predictions(model: nn.Module, clips: list, downsample: np.ndarray, 
                                 img_size: tuple, device: str, args: dict, output_dir: str) -> np.ndarray:
    """
    Run model inference to convert 2D keypoints into 3D keypoints and standardize them.
    Does NOT create or save any images.

    Args:
        model (nn.Module): Trained MotionAGFormer model.
        clips (list): List of keypoint clips.
        downsample (np.ndarray): Downsampling indices.
        img_size (tuple): Image dimensions (height, width, channels).
        device (str): Device to run inference on.
        args (dict): Model configuration arguments.
        output_dir (str): Directory where output will be saved.

    Returns:
        np.ndarray: Array of all standardized 3D poses, shape (num_frames, num_joints, 3).
    """
    print('\nGenerating 3D pose (inference only)...')
    all_poses = []
    for idx, clip in tqdm(enumerate(clips)):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 
        input_2D_aug = flip_data(input_2D)

        input_2D = torch.from_numpy(input_2D.astype('float32')).to(device)
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).to(device)

        output_3D_non_flip = model(input_2D) 
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()

        for post_out in post_out_all:
            post_out_std = standardize_3d_keypoints(post_out)
            all_poses.append(post_out_std)

    print('3D pose inference successful!')
    return np.stack(all_poses, axis=0)


def standardize_3d_keypoints(keypoints: np.ndarray) -> np.ndarray:
    rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    keypoints = camera_to_world(keypoints, R=rot, t=0)
    keypoints[:, 2] -= np.min(keypoints[:, 2])
    max_value = np.max(keypoints)
    keypoints /= max_value
    return keypoints


def visualize_3D_poses(poses: np.ndarray, output_dir: str, pro_video_file: np.ndarray) -> None:
    """Visualize and save 3D pose images from standardized 3D keypoints.

    Args:
        poses (np.ndarray): Array of standardized 3D poses, shape (num_frames, num_joints, 3).
        output_dir (str): Directory where output will be saved.
    """
    output_dir_3D = os.path.join(output_dir, 'pose3D')
    os.makedirs(output_dir_3D, exist_ok=True)

    print('\nGenerating 3D pose images...')
    for idx, post_out_std in enumerate(tqdm(poses)):
        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')
        # show3Dpose(post_out_std, ax)

        output_path_3D = os.path.join(output_dir_3D, f"{idx:04d}_3D.png")
        if pro_video_file is not None:
            pro_post_out_std = standardize_3d_keypoints(pro_video_file[idx])
            create_pose_overlay_image(post_out_std, pro_post_out_std, output_path_3D)
        plt.savefig(output_path_3D, dpi=200, format='png', bbox_inches='tight')
        plt.close(fig)

    print('3D pose visualization successful!')



def create_pose_overlay_image(data1: np.ndarray, data2: np.ndarray, output_path: str):
    """Create a single image with 3D pose keypoints from data1 and data2 overlaid on the same axis.
    Both data1 and data2 are standardized before plotting.

    Args:
        data1 (np.ndarray): Shape (num_joints, 3)
        data2 (np.ndarray): Shape (num_joints, 3)
        output_path (str): Output path for the image (e.g., .png)
    """
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0], projection='3d')
    
    std_data1 = standardize_3d_keypoints(data1)
    std_data2 = standardize_3d_keypoints(data2)
    
    show3Dpose(std_data1, ax)
    show3Dpose(std_data2, ax)

    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Overlay image saved to {output_path}")
    return output_path


def _generate_demo_images(output_dir_2D: str, output_dir_3D: str, output_dir: str) -> None:
    """Generate combined demo images showing 2D input and 3D reconstruction side by side.
    
    Args:
        output_dir_2D (str): Directory containing 2D pose images.
        output_dir_3D (str): Directory containing 3D pose images.
        output_dir (str): Base output directory.
    """
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    output_dir_pose = os.path.join(output_dir, 'pose')
    os.makedirs(output_dir_pose, exist_ok=True)

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        # Crop images
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        # Create combined visualization
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize=font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize=font_size)

        # Save combined image
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        output_path_pose = os.path.join(output_dir_pose, f"{i:04d}_pose.png")
        plt.savefig(output_path_pose, dpi=200, bbox_inches='tight')
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