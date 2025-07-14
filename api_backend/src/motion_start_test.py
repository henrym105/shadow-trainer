import numpy as np

def find_motion_start(keypoints: np.ndarray, is_lefty: bool = True, min_pct_change: float = 5) -> int:
    """
    Find the first frame where the distance between the front foot and sacrum (index 0)
    changes by more than min_pct_change percent relative to the distance in the first frame, using only the z direction.

    Args:
        keypoints (np.ndarray): Shape (n_frames, 17, 3)
        is_lefty (bool): If True, use left foot (index 6) as front foot; else right foot (index 3).
        min_pct_change (float): Minimum relative decrease (as percent) to count as movement.

    Returns:
        int: Index of the first frame where movement is detected.
    """
    # right_ankle = 3, left_ankle = 6
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

        if pct_change > min_pct_change:
            return i

    return 0  # fallback: no movement detected



if __name__ == "__main__":
    # Example usage
    # keypoints = np.load('/home/ec2-user/shadow-trainer/api_backend/tmp_api_output/32a19f6e-13f0-488e-85fc-b5dc9407d85c_output/raw_keypoints/2D_keypoints.npy')
    keypoints = np.load('/home/ec2-user/shadow-trainer/api_backend/tmp_api_output/1fbfb7a2-6c44-46be-ac9f-2d29a5f1b8dd_output/raw_keypoints/user_3D_keypoints.npy')
    # keypoints = np.load('api_backend/checkpoint/example_SnellBlake.npy')
    start_frame = find_motion_start(keypoints, is_lefty=False, min_pct_change=3)
    print(f"Motion starts at frame: {start_frame}")