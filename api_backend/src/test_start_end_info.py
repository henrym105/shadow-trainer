# test_start_end_info.py

import numpy as np
import matplotlib.pyplot as plt


def get_start_end_info(arr, is_lefty: bool = False) -> tuple:
    """Determine the start and end points of the motion based on the ankle positions in the 3D keypoints array.

    Args:
        arr (np.ndarray): Input numpy array, assumed to be of shape (n_frames, 17, 3) for 3D keypoints.

    Returns:
        tuple: A tuple containing (min_value, max_value, average_value).
    """
    assert arr.ndim == 3, "Input array must be 3D with shape (n_frames, 17, 3). Received shape: {}".format(arr.shape)
    print(f"[DEBUG] get_start_end_info() - Input array shape: {arr.shape}")

    # left_ankle_arr = []
    # right_ankle_arr = []
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

        # print(i, max_front_index, front_ankle_z)
        print(f"[DEBUG] Frame {i}: {front_ankle_name} Z: {front_ankle_z}, {back_ankle_name} Z: {back_ankle_z}")
        
        if front_ankle_z < 0.005:
            print(f"    ----> Frame {i} - Low point detected at {front_ankle_name} Z: {front_ankle_z}")
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

    print(f"Switch point: {switch_point}")
    # is_valid = True
    
    # # if less than 80% of the values before switch_point in higher_ankle_arr are 1 print invalid
    # if np.sum(higher_ankle_arr[:switch_point]) < len(higher_ankle_arr[:switch_point])*0.8:
    #     is_valid = False
    # # if less than 80% of the values after switch_point in higher_ankle_arr are 0 print invalid
    # if np.sum(higher_ankle_arr[switch_point:]) > len(higher_ankle_arr[switch_point:])*0.2:
    #     is_valid = False

    # # if is_valid:
    # #     print("Valid")

    end_point = switch_point

    # plot the joints
    plt.plot(front_ankle_arr, label = front_ankle_name + " (Front Ankle)")
    plt.plot(back_ankle_arr, label = back_ankle_name + " (Back Ankle)")
    plt.axvline(x=starting_point, color='r', linestyle='--', label=f'Starting Point (low point) {starting_point}')
    plt.axvline(x=end_point, color='g', linestyle='--', label=f'End Point (switch point) {end_point}')
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=2)
    plt.show()
    plt.savefig("ankle_plot.png", dpi=150, bbox_inches='tight')
    plt.close()

    for i in range(switch_point+5, len(back_ankle_arr)):
        if back_ankle_arr[i] < 0.005:
            end_point = i
            break
    
    return (starting_point, end_point)


def get_frame_info(frame: np.ndarray) -> dict:
    """Get the joint names and their corresponding coordinates from a single frame of 3D key points.

    Args:
        frame (np.ndarray): A single frame of 3D key points, expected shape (
            17, 3), where each row corresponds to a joint and each column corresponds to x, y, z coordinates.

    Returns:
        dict: A dictionary mapping joint names to their coordinates.
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
    # print(f"\n[DEBUG] get_frame_info() - Extracted joints from frame: {joints.keys()}")
    # pprint(joints)
    return joints



if __name__ == "__main__":
    # Example usage
    # Assuming arr is a numpy array of shape (n_frames, 17, 3)

    # Create a dummy array for testing
    # n_frames = 100
    # arr = np.random.rand(n_frames, 17, 3)

    # path = '/home/ec2-user/shadow-trainer/api_backend/tmp_api_output/6910b3de-95d3-4978-9375-673bed3dc93b_output/raw_keypoints/user_3D_keypoints.npy'
    path = '/home/ec2-user/shadow-trainer/api_backend/tmp_api_output/e88a6f27-9d9e-4ce7-a5c4-366f8c3b755e_output/raw_keypoints/user_3D_keypoints.npy'
    arr = np.load(path)

    # Call the function with the dummy array
    start_end_info = get_start_end_info(arr, is_lefty=False)
    print(f"Start and End Points: {start_end_info}")