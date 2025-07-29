import os
import logging

import cv2
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from celery.utils.log import get_task_logger

from constants import API_ROOT_DIR, CHECKPOINT_DIR
from src.utils import get_pytorch_device, get_frame_info

logger = get_task_logger(__name__)


class YOLOPoseEstimator:
    def __init__(self, model_name="yolo11x-pose.pt", checkpoint_dir=CHECKPOINT_DIR, device="cpu"):
        """Initializes the YOLOPoseEstimator with a specified YOLOv11 pose model for 2d keypoint detection.

        Args:
            model_name (str): The name of the YOLOv11 pose model to use.
            checkpoint_dir (str): Local directory where the YOLO model object will be stored.
        """
        model_path = os.path.join(checkpoint_dir, model_name)
        self.model = YOLO(model_path)
        self.model.to(device)


    def get_keypoints(self, frame) -> np.ndarray:
        """Processes a single video frame using the YOLOv11 pose model and
        returns keypoints in the hr_keypoints format.

        Args:
            frame (np.ndarray): The input frame (image) as a NumPy array.

        Returns:
            np.ndarray: A NumPy array of shape (17, 3) where N is the number of hr_keypoints and each row is [x, y, confidence].
        """
        if frame is None:
            raise ValueError("Frame is None, cannot process keypoints")
            
        pose_results = self.model(frame, verbose=False)
        keypoints = None
        
        if pose_results and pose_results[0].keypoints and len(pose_results[0].keypoints.xyn) > 0:
            # Get keypoints of the first detected pose as numpy array
            keypoints_xy = pose_results[0].keypoints.xyn[0].cpu().numpy()  # shape (N, 2)
            keypoints_conf = pose_results[0].keypoints.conf[0].cpu().numpy()  # shape (N,)
            keypoints = np.concatenate([keypoints_xy, keypoints_conf[:, None]], axis=1)  # shape (N, 3)

        if keypoints is None:
            raise ValueError("No keypoints detected in frame")
            
        assert keypoints.shape == (17, 3), "Keypoints should have shape (17, 3) with format (x, y, conf). Received shape: {}".format(keypoints.shape)

        return keypoints


    def get_points_from_frame(self, frame):
        """
        Processes a single video frame using the YOLOv11 pose model and
        returns keypoints in the hr_keypoints format.

        Args:
            frame (np.ndarray): The input frame (image) as a NumPy array.

        Returns:
            np.ndarray or None: A NumPy array of shape (N, 3) where N is the number
                                of hr_keypoints and each row is [x, y, confidence],
                                or None if no poses are detected.
        """
        if frame is None:
            logger.error("Frame is None, cannot process keypoints")
            return None
            
        try:
            keypoints = self.get_keypoints(frame)
        except (ValueError, AssertionError) as e:
            logger.error(f"Failed to get keypoints from frame: {e}")
            return None
        thorax = ((keypoints[5] + keypoints[6])/2)
        mid_hip = ((keypoints[12] + keypoints[11])/2)
        spine = (mid_hip+thorax)/2
        mid_head = ((keypoints[3] + keypoints[4])/2)

        keypoints = np.insert(keypoints, 0, mid_hip, axis=0)
        keypoints = np.insert(keypoints, 0, spine, axis=0)
        keypoints = np.insert(keypoints, 0, thorax, axis=0)
        keypoints = np.insert(keypoints, 0, mid_head, axis=0)
        #print what type keypoints is
        hr_keypoints = np.array([keypoints[14]])
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[12], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[10], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[13], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[11], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[9], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[0], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, (keypoints[0]+keypoints[1])/2, axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[1], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[2], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[19], axis=0)

        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[17], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[15], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[20], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[18], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[16], axis=0)
        hr_keypoints = np.insert(hr_keypoints, 0, keypoints[3], axis=0)

        #multiply first column by frame_width
        hr_keypoints[:, 0] *= frame.shape[1]
        #multiply second column by frame_height
        hr_keypoints[:, 1] *= frame.shape[0]

        # print(" [get_points_from_frame] Keypoints shape:", hr_keypoints.shape)
        # print(" [get_points_from_frame] sample keypoint 0:", hr_keypoints[0])
        return hr_keypoints


    def get_keypoints_from_video(self, video_path):
        """Processes a video file to extract keypoints for each frame.

        Args:
            video_path (str): Path to the input video file.
        
        Returns:
            np.ndarray: An array with shape (1, num_frames, 17, 3) where num_frames is the number of frames in the video.
                        Each element is a NumPy array of shape (17, 3) representing the person's keypoints in hr_keypoints format.
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keypoints_array = np.empty(shape = (1, frame_count, 17, 3))

        for i in tqdm(range(frame_count), desc="Extracting 2D keypoints from input video"):
            ret, frame = cap.read()
            if not ret:
                break
            new_keypoints = self.get_points_from_frame(frame)
            keypoints_array[0][i] = new_keypoints
        cap.release()

        return keypoints_array


def rotate_video_until_upright(video_path: str, debug: bool = False) -> int:
    """Detect if a video is properly oriented based on human pose keypoints.
    If it's not, rotate it to be right-side-up, overwriting the original video file.

    Args:
        video_path (str): Path to the input video file

    Returns:
        int: how many 90 degree clockwise rotations were applied to make the video upright.
    """
    logger.info(f"Analyzing video orientation for: {video_path}")
    
    clockwise_rotations_needed = 0
    device = get_pytorch_device()
    yoloEstimator = YOLOPoseEstimator(device=device)
    
    # Get the first frame, use it to determine rotation needed to make the person upright
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    keypoints = yoloEstimator.get_points_from_frame(frame)
    
    if keypoints is None:
        logger.warning(f"No keypoints detected in first frame of {video_path}. Assuming video is upright.")
        cap.release()
        return 0
    
    is_upright = is_person_upright(keypoints)

    while not is_upright and clockwise_rotations_needed < 4:
        logger.info(f"Incorrect Orientation Detected. Rotating {clockwise_rotations_needed * 90} degrees and check again...")
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        clockwise_rotations_needed += 1
        keypoints = yoloEstimator.get_points_from_frame(frame)
        
        if keypoints is None:
            logger.warning(f"No keypoints detected after rotation. Assuming current orientation is correct.")
            break
            
        is_upright = is_person_upright(keypoints)
    cap.release()
    assert not cap.isOpened()
    if clockwise_rotations_needed == 0:
        # No rotation needed, video is already upright, break early
        logger.info("Video is already upright. No rotation needed.")
        return 0

    # Now rotate the entire video to match the first frame's orientation
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine output size based on number of rotations
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if clockwise_rotations_needed % 2 == 0:
        # 0 or 180 degrees - dimensions stay the same
        size = (original_width, original_height)
    else:
        # 90 or 270 degrees - dimensions swap
        size = (original_height, original_width)

    output_path = video_path
    if debug:
        output_path = os.path.join(os.path.dirname(video_path), "rotated_" + os.path.basename(video_path))
    if debug: logger.info(f"fourcc: {fourcc}, fps: {fps}, size: {size}, output_path: {output_path}")

    output_cap = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=size)

    # Map rotation count to cv2 rotation constants
    rotation_map = {
        0: None,
        1: cv2.ROTATE_90_CLOCKWISE,
        2: cv2.ROTATE_180,
        3: cv2.ROTATE_90_COUNTERCLOCKWISE
    }
    rotation_constant = rotation_map[clockwise_rotations_needed % 4]

    for i in tqdm(range(frame_count), desc=f"Rotating Video [{clockwise_rotations_needed * 90} degrees]"):
        ret, frame = cap.read()
        if not ret:
            break
        else:
            if (debug and i == 0): cv2.imwrite("debug_first_frame_original.jpg", frame)
            if rotation_constant is not None:
                frame = cv2.rotate(frame, rotation_constant)
            output_cap.write(frame)
        
            if (debug and i == 0): cv2.imwrite("debug_first_frame_rotated.jpg", frame)

    cap.release()
    output_cap.release()
    logger.info(f"Video orientation corrected. Saved to: {video_path}")

    return clockwise_rotations_needed



def is_person_upright(keypoints: np.ndarray) -> bool:
    """Determines if the person in the keypoints is upright based on head position. 
    Assumes y-axis is vertical with higher values being lower on the image. (0,0) is top-left corner

    Args:
        keypoints (np.ndarray): Keypoints of shape (17, 3) where each row is [x, y, confidence].

    Returns:
        bool: True if the person is upright, False otherwise.
    """
    logger.info(f"Keypoints shape: {keypoints.shape}")
    joints = get_frame_info(keypoints)
    
    # joints will be like {'Hip': [x, y, z], 'Left Shoulder': [x, y, z], ...}
    head_y = joints['Head'][1]
    shoulders_y = joints['Left Shoulder'][1]
    hips_y = joints['Right Hip'][1]
    knees_y = joints['Left Knee'][1]
    feet_y = joints['Right Ankle'][1]
    logger.info(f"Head Y: {int(head_y)}, Shoulders Y: {int(shoulders_y)}, Hips Y: {int(hips_y)}, Knees Y: {int(knees_y)}, Feet Y: {int(feet_y)}")

    return (head_y < shoulders_y < hips_y < knees_y < feet_y)



def mirror_video_for_lefty(video_path: str, debug: bool = False) -> str:
    """Mirrors the video horizontally to simulate a left-handed view.

    Args:
        video_path (str): Path to the input video file.
        debug (bool): If True, saves a debug copy of the mirrored video.

    Returns:
        str: Path to the mirrored video file.
    """
    logger.info(f"Mirroring video for left-handed view: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Always use a temporary file to avoid overwriting the input while reading
    temp_output_path = os.path.join(os.path.dirname(video_path), "temp_mirrored_" + os.path.basename(video_path))
    
    output_cap = cv2.VideoWriter(filename=temp_output_path, fourcc=fourcc, fps=fps, frameSize=(original_width, original_height))

    for _ in tqdm(range(frame_count), desc="Mirroring Video"):
        ret, frame = cap.read()
        if not ret:
            break
        mirrored_frame = cv2.flip(frame, 1)  # Flip horizontally
        output_cap.write(mirrored_frame)

    cap.release()
    output_cap.release()
    
    # Determine final output path
    if debug:
        final_output_path = os.path.join(os.path.dirname(video_path), "mirrored_" + os.path.basename(video_path))
        os.rename(temp_output_path, final_output_path)
    else:
        final_output_path = video_path
        os.replace(temp_output_path, video_path)
    
    logger.info(f"Video mirrored for left-handed view. Saved to: {final_output_path}")
    
    return final_output_path


if __name__ == "__main__":    
    # Example usage
    import torch
    estimator = YOLOPoseEstimator(
        checkpoint_dir=os.path.join(API_ROOT_DIR, "checkpoint"),
        model_name="yolo11n-pose.pt",
        device= "mps" if torch.backends.mps.is_available() else "cpu"
    )
    test_video = "/Users/Henry/github/shadow-trainer/api_backend/tmp_api_output/henry-mini.mov"
    keypoints = estimator.get_keypoints_from_video(test_video)
    logger.info(f"Number of frames processed: {len(keypoints)}")
    logger.info(f"Keypoints shape first frame: {keypoints[0].shape if keypoints[0] is not None else 'No keypoints detected'}")

    # logger.info(f"Keypoints shape: {keypoints.shape}")

    # Save the keypoints to a file or process them further as needed
    # For example, you can save them to a .npy file:
    output_file = os.path.join(API_ROOT_DIR, "tmp_api_output", "output_keypoints_2d_yolov11.npy")
   
    # np.save(output_file, keypoints)
    # logger.info(f"Keypoints saved to {output_file}")

    pass
