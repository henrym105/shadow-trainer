import os
import numpy as np
import boto3
import tempfile


def time_warp_pro_video(amateur_data: np.ndarray, professional: np.ndarray):
    """    Time warps the professional data to align with the amateur data.
    Args:
        amateur_data (np.ndarray): The amateur data as a NumPy array.
        professional (np.ndarray): The professional data as a NumPy array.
    Returns:
        np.ndarray: The time-warped professional data. Shape will match the amateur data.
    """
    valid, switch_point, max_y_pt, ankle_points = get_numpy_info(amateur_data)
    print(valid, switch_point, max_y_pt)

    professional_kpts = shift_data_time(professional, 200, 100, max_y_pt, switch_point - max_y_pt, len(amateur_data) - switch_point)
    
    shapes_match = (professional_kpts.shape[0] == amateur_data.shape[0])
    assert shapes_match, f"Professional keypoints array shape {professional_kpts.shape} does not match amateur data shape {amateur_data.shape}"
    return professional_kpts



def get_numpy_info(arr):
  """
  Gets the minimum, maximum, and average values from a numpy array.

  Args:
    arr (np.ndarray): Input numpy array.

  Returns:
    tuple: A tuple containing (min_value, max_value, average_value).
  """

  # Robust version: handle short arrays and edge cases
  n_frames = len(arr)
  if n_frames < 15:
      # Not enough frames for smoothing or switch logic
      print("[get_numpy_info] Too few frames for robust analysis, skipping temporal alignment.")
      return (False, 0, 0, [0,0,0,0])

  left_ankle_arr = []
  right_ankle_arr = []
  higher_ankle_arr = []
  head_arr = []
  max_left = 0
  max_left_index = 0
  arm_movement = [0]
  try:
      prev_arm_position = get_frame_info(arr[0])["Right Wrist"]
  except Exception as e:
      print(f"[get_numpy_info] Error extracting Right Wrist: {e}")
      return (False, 0, 0, [0,0,0,0])

  is_valid = True

  for i in range(n_frames):
      try:
          joints = get_frame_info(arr[i])
          left_ankle_z = joints["Left Ankle"][2]
          right_ankle_z = joints["Right Ankle"][2]
          head_z = joints["Head"][2]
          arm_movement.append(np.linalg.norm(np.array(joints["Right Wrist"]) - np.array(prev_arm_position)))
          if left_ankle_z > max_left:
              max_left = left_ankle_z
              max_left_index = i
          left_ankle_arr.append(left_ankle_z)
          right_ankle_arr.append(right_ankle_z)
          head_arr.append(head_z)
      except Exception as e:
          print(f"[get_numpy_info] Error at frame {i}: {e}")
          return (False, 0, 0, [0,0,0,0])

  # Smoothing
  try:
      left_ankle_arr = np.array(left_ankle_arr)
      right_ankle_arr = np.array(right_ankle_arr)
      left_ankle_arr = np.convolve(left_ankle_arr, np.ones(10)/10, mode='valid')
      right_ankle_arr = np.convolve(right_ankle_arr, np.ones(10)/10, mode='valid')
  except Exception as e:
      print(f"[get_numpy_info] Error in smoothing: {e}")
      return (False, 0, 0, [0,0,0,0])

  for i in range(len(left_ankle_arr)):
      if left_ankle_arr[i] > right_ankle_arr[i]:
          higher_ankle_arr.append(1)
      else:
          higher_ankle_arr.append(0)

  # Switch point logic
  switch_point = 0
  for i in range(max_left_index, len(higher_ankle_arr)):
      if left_ankle_arr[i] <= 0.003:
          switch_point = i
          break
  # Validity checks
  if switch_point == 0 or switch_point >= len(higher_ankle_arr):
      print("[get_numpy_info] No valid switch point found.")
      return (False, 0, max_left_index, [0,0,0,0])
  if np.sum(higher_ankle_arr[:switch_point]) < len(higher_ankle_arr[:switch_point])*0.8:
      is_valid = False
  if np.sum(higher_ankle_arr[switch_point:]) > len(higher_ankle_arr[switch_point:])*0.2:
      is_valid = False
  if max(arm_movement) < 0.37:
      is_valid = False

  ankle_points = [0,0,0,0]
  try:
      switch_joints = get_frame_info(arr[switch_point])
      ankle_points[0] = switch_joints["Right Ankle"][0]
      ankle_points[1] = switch_joints["Right Ankle"][1]
      ankle_points[2] = switch_joints["Left Ankle"][0]
      ankle_points[3] = switch_joints["Left Ankle"][1]
  except Exception as e:
      print(f"[get_numpy_info] Error extracting ankle points: {e}")
      return (False, switch_point, max_left_index, [0,0,0,0])

  if np.linalg.norm(np.array(ankle_points[:2]) - np.array(ankle_points[2:])) <= 0.1:
      is_valid = False

  print(f"[get_numpy_info] is_valid={is_valid}, switch_point={switch_point}, max_left_index={max_left_index}, ankle_points={ankle_points}")
  return (is_valid, switch_point, max_left_index, ankle_points)

def get_frame_info(frame):
  joint_names = [
    "Hip", "Right Hip", "Right Knee", "Right Ankle",
    "Left Hip", "Left Knee", "Left Ankle", "Spine",
    "Thorax", "Neck", "Head", "Left Shoulder",
    "Left Elbow", "Left Wrist", "Right Shoulder",
    "Right Elbow", "Right Wrist"
    ]
  #frame is of shape 17, 3 for each joint print the x, y and z coordinates
  joints = {}
  for i in range(len(frame)):
    #print(f"{joint_names[i]}: {frame[i]}")
    joints[joint_names[i]] = frame[i]
  return joints



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


# prompt: Create a function that takes in 2 x,y coordinates and finds what angle they are pointing
def find_angle(coord1, coord2):
  """
  Finds the angle in degrees between the line segment connecting two 2D coordinates
  and the positive x-axis.

  Args:
    coord1 (tuple): A tuple representing the first (x, y) coordinate.
    coord2 (tuple): A tuple representing the second (x, y) coordinate.

  Returns:
    float: The angle in degrees. Returns None if the points are the same.
  """
  x1, y1 = coord1
  x2, y2 = coord2

  print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

  dx = x2 - x1
  dy = y2 - y1

  if dx == 0 and dy == 0:
    return None # Points are the same, angle is undefined

  angle = np.degrees(np.arctan2(dy, dx))
  if angle < 0:
    angle += 360
  print(f"Angle: {angle}")
  return angle



def resample_pose_sequence(pose_seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a pose sequence to have exactly `target_len` frames using linear interpolation.

    Args:
        pose_seq (np.ndarray): Input array of shape (n, 17, 3), where n is the number of frames.
        target_len (int): Desired number of output frames.

    Returns:
        np.ndarray: Resampled array of shape (target_len, 17, 3)
    """
    n_frames, n_joints, n_dims = pose_seq.shape
    if n_frames == target_len:
        return pose_seq.copy()

    # Create an array of indices for the original and target frame positions
    original_indices = np.linspace(0, n_frames - 1, num=n_frames)
    target_indices = np.linspace(0, n_frames - 1, num=target_len)

    # Interpolate along the time axis for each joint and dimension
    resampled = np.zeros((target_len, n_joints, n_dims), dtype=np.float32)
    for j in range(n_joints):
        for d in range(n_dims):
            resampled[:, j, d] = np.interp(target_indices, original_indices, pose_seq[:, j, d])

    return resampled


def recenter_on_left_ankle(pose_array, target_xyz=(0.0, 0.0, 0.0)):
    """Recenter each frame so that the left ankle (joint 6) moves to the given target coordinates.

    Parameters:
    - pose_array (np.ndarray): Shape (n, 17, 3)
    - target_xyz (tuple or list or np.ndarray): The target (x, y, z) coordinates to move the left ankle to.

    Returns:
    - np.ndarray: Recentered pose array
    """
    left_ankle = pose_array[:, 6:7, :]  # shape (n, 1, 3)
    target = np.array(target_xyz).reshape(1, 1, 3)  # shape (1, 1, 3) for broadcasting
    print(target)
    return (pose_array - left_ankle + target)


def recenter_on_right_ankle(pose_array):
    """Recenter each frame around the right ankle (joint 3).

    Parameters:
    - pose_array (np.ndarray): Shape (n, 17, 3)

    Returns:
    - np.ndarray: Recentered pose array
    """
    right_ankle = pose_array[:, 3:4, :]  # shape (n, 1, 3) for broadcasting
    return pose_array - right_ankle


def shift_data_time(data, switch, max_y, time_1=100, time_2=100, time_3=150):
    segment_1 = data[:max_y,:,:]
    segment_2 = data[max_y: switch,:,:]
    segment_3 = data[switch-1:,:,:]
    segment_1 = resample_pose_sequence(segment_1, time_1)
    segment_2 = resample_pose_sequence(segment_2, time_2)
    left_ankle_last = segment_2[-1,6]
    segment_3 = resample_pose_sequence(segment_3, time_3)
    final_arr = np.concatenate([segment_1,segment_2,segment_3], axis=0)
    print(final_arr.shape)
    return final_arr



def get_s3_client(aws_access_key_id=None, aws_secret_access_key=None, region_name='us-east-2'):
    if aws_access_key_id and aws_secret_access_key:
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
    else:
        return boto3.client('s3')

def list_s3_files(s3, bucket_name, prefix, filetype='.npy'):
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith(filetype) and 'FF' not in key:
                    yield key


def download_s3_file(s3, bucket_name, key, filetype):
    with tempfile.NamedTemporaryFile(delete=False, suffix=filetype) as temp_file:
        temp_path = temp_file.name
    s3.download_file(bucket_name, key, temp_path)
    return temp_path

def process_pose_file(data):
    valid, switch_point, max_y_pt, ankle_points = get_numpy_info(data)
    angle = find_angle(ankle_points[0:2], ankle_points[2:4])
    if valid:
        for i in range(len(data)):
            data[i] = rotate_along_z(data[i], 135 - angle)
        data = shift_data_time(data, switch_point, max_y_pt)
    return valid, data, switch_point, max_y_pt, ankle_points, angle

def build_save_path(key, filetype):
    base_name = os.path.splitext(os.path.basename(key))[0]
    path_name = os.path.dirname(key)
    path_name = path_name.split('/')[1:]
    path_name = '/'.join(path_name)
    save_path = f"cleaned_numpy/{path_name}/{base_name}{filetype}"
    return base_name, save_path

def handle_pose_file(s3, bucket_name, key, filetype, dryrun, prev_frames, cleaned_list, val_list):
    print(f"\n[INFO] Processing: {key}")
    if dryrun:
        print(f"[DRYRUN] Would download: {key}")
        return prev_frames
    try:
        temp_path = download_s3_file(s3, bucket_name, key, filetype)
        data = np.load(temp_path)
        print(data.shape)
        valid, data, switch_point, max_y_pt, ankle_points, angle = process_pose_file(data)
        val_list.append(valid)
        base_name, save_path = build_save_path(key, filetype)
        print(f"Path name: {os.path.dirname(key)}")
        print(f"Base name: {base_name}")
        print(f"Save path: {save_path}")
    except Exception as e:
        print("FAILED", e)
    return prev_frames


def list_and_play_mp4_from_s3(
    bucket_name, 
    prefix, 
    aws_access_key_id=None, 
    aws_secret_access_key=None, 
    region_name='us-east-2',
    filetype='.npy',
    dryrun=True
):
    s3 = get_s3_client(aws_access_key_id, aws_secret_access_key, region_name)
    cleaned_list = []
    val_list = []
    prev_frames = None
    for key in list_s3_files(s3, bucket_name, prefix, filetype):
        if dryrun:
            print(f"[DRYRUN] Would process: {key}")
            continue
        else:
            prev_frames = handle_pose_file(
                s3, bucket_name, key, filetype, dryrun, prev_frames, cleaned_list, val_list
            )
    print(val_list)
    if len(val_list) > 0:
        print(np.sum(val_list)/len(val_list))
    return cleaned_list


# def find_all_pro_npy_files(download: bool = False):
#     pass


if __name__ == "__main__":
    # Example usage
    bucket_name = "shadow-trainer-prod"
    prefix = "pro_3d_keypoints"
    # cleaned_data = find_all_pro_npy_files(bucket_name, prefix, dryrun=True)
    # print(f"Processed {len(cleaned_data)} valid pose sequences.")