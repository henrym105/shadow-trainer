import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import os

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class YOLOPoseEstimator:
    def __init__(self, checkpoint_dir=".", model_name="yolo11x-pose.pt", device="cpu"):
        """Initializes the YOLOPoseEstimator with a specified YOLOv11 pose model.

        Args:
            model_name (str): The name of the YOLOv11 pose model to use.
            checkpoint_dir (str): Local directory where the YOLO model object will be stored.
        """
        model_path = os.path.join(checkpoint_dir, model_name)
        self.model = YOLO(model_path)
        self.model.to(device)
        self.yolo_to_hr_map = {
            15: 0,  # right_ankle -> R_Ankle
            14: 1,  # right_knee -> R_Knee
            12: 2,  # right_hip -> R_Hip
            11: 3,  # left_hip -> L_Hip
            13: 4,  # left_knee -> L_Knee
            16: 5,  # left_ankle -> L_Ankle
            10: 6,  # right_wrist -> R_Wrist
            8: 7,   # right_elbow -> R_Elbow
            6: 8,   # right_shoulder -> R_Shoulder
            5: 9,   # left_shoulder -> L_Shoulder
            7: 10,  # left_elbow -> L_Elbow
            9: 11,  # left_wrist -> L_Wrist
            # Neck: typically derived from shoulders/ears
            # Head_top: typically derived from nose/eyes/ears
        }


    def get_keypoints(self, frame) -> np.ndarray:
        """Processes a single video frame using the YOLOv11 pose model and
        returns keypoints in the hr_keypoints format.

        Args:
            frame (np.ndarray): The input frame (image) as a NumPy array.

        Returns:
            np.ndarray: A NumPy array of shape (17, 3) where N is the number of hr_keypoints and each row is [x, y, confidence].
        """
        pose_results = self.model(frame, verbose=False)
        if pose_results and pose_results[0].keypoints:
            # Get keypoints of the first detected pose as numpy array
            keypoints_xy = pose_results[0].keypoints.xyn[0].cpu().numpy()  # shape (N, 2)
            keypoints_conf = pose_results[0].keypoints.conf[0].cpu().numpy()  # shape (N,)
            keypoints = np.concatenate([keypoints_xy, keypoints_conf[:, None]], axis=1)  # shape (N, 3)

        # Add a new first dimension to keypoints for compatability (e.g., (num_frames, 17, 3) -> (1, num_frames, 17, 3))
        # Previous code accounts for multiple people could be detected in each frame
        # keypoints = keypoints[None, ...]

        # Reorder keypoints according to yolo_to_hr_map to match hr_keypoints format
        # hr_keypoints = np.zeros((17, 3), dtype=keypoints.dtype)
        # for yolo_idx, hr_idx in self.yolo_to_hr_map.items():
        #     hr_keypoints[hr_idx] = keypoints[yolo_idx]
        # keypoints = hr_keypoints

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
        keypoints = self.get_keypoints(frame)
        thorax = ((keypoints[5] + keypoints[6])/2)
        mid_hip = ((keypoints[12] + keypoints[11])/2)
        spine = (mid_hip+thorax)/2
        mid_head = ((keypoints[3] + keypoints[4])/2)

        joint_names = [
            "Hip", "Right Hip", "Right Knee", "Right Ankle",
            "Left Hip", "Left Knee", "Left Ankle", "Spine",
            "Thorax", "Neck", "Head", "Left Shoulder",
            "Left Elbow", "Left Wrist", "Right Shoulder",
            "Right Elbow", "Right Wrist"
        ]
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
        #add an extra value to each part of hr_keypoint axis 1
        # hr_keypoints = np.insert(hr_keypoints, 2, 0.8, axis=1)

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
        keypoints_list = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for _ in tqdm(range(frame_count), desc="Extracting 2D keypoints from input video"):
            ret, frame = cap.read()
            if not ret:
                break
            new_keypoints = self.get_points_from_frame(frame)
            keypoints_list.append(new_keypoints)
        cap.release()

        keypoints_array = np.array(keypoints_list, dtype=object)
        
        # Add a new first dimension to keypoints for compatability (e.g., (num_frames, 17, 3) -> (1, num_frames, 17, 3))
        # Previous code accounts for multiple people could be detected in each frame
        keypoints_array = keypoints_array[None, ...]

        return keypoints_array


if __name__ == "__main__":    
    # Example usage
    import torch
    estimator = YOLOPoseEstimator(
        checkpoint_dir=os.path.join(BACKEND_ROOT, "checkpoint"),
        model_name="yolo11n-pose.pt",
        device= "mps" if torch.backends.mps.is_available() else "cpu"
    )
    test_video = "/Users/Henry/github/shadow-trainer/api_backend/tmp_api_output/henry-mini.mov"
    keypoints = estimator.get_keypoints_from_video(test_video)
    print("Number of frames processed:", len(keypoints))
    print("Keypoints shape first frame:", keypoints[0].shape if keypoints[0] is not None else "No keypoints detected")

    print(keypoints.shape)

    # Save the keypoints to a file or process them further as needed
    # For example, you can save them to a .npy file:
    output_file = os.path.join(BACKEND_ROOT, "tmp_api_output", "output_keypoints_2d_yolov11.npy")
    
    
    np.save(output_file, keypoints)
    print(f"Keypoints saved to {output_file}")