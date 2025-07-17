from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

IS_LEFTY = True



# Load the YOLO11s-pose model and move to CUDA if available
model = YOLO('yolo11x-pose.pt')
if hasattr(model, 'to'):
    model.to('cuda')



def get_recovery_point(rankle_y_diff):
    # Return the recovery point, which is the index of the first occurrence of rankle_y_diff > 0.5
    recovery_point = -1
    for i in range(len(rankle_y_diff)):
        if rankle_y_diff[i] > 0.5:
            recovery_point = i
            break
    return recovery_point


def get_keypoints(keypoints):
    #get the save thre right wrist, right ankle and left ankle in an array of shape (3, 3) with the x, y, and confidence
    if keypoints is None:
        return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    keypoints = keypoints.cpu().numpy()
    min_leg = 100000
    count = 0
    for joints in keypoints.xy:
        count += 1
        # print(f"Processing keypoint {count}: {joints}")
        if joints[16][1] < min_leg:
            min_leg = joints[16][1]
            right_wrist = joints[10]
            right_ankle = joints[15]
            left_ankle = joints[16]
    # print(f"Right Wrist: {right_wrist}, Right Ankle: {right_ankle}, Left Ankle: {left_ankle}")
    return np.array([right_wrist, right_ankle, left_ankle])


def pose_and_overlay_video(input_video_path):
    points_by_frame = []
    cap = cv2.VideoCapture(input_video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if IS_LEFTY:
            print(frame.shape)
            frame = flip_video_frames(frame)
            print(frame.shape)



        results = model(frame, device=0)  # Force CUDA (device=0)
        # get the pose points
        # print(results)
        points_for_frame = []
        max_leg = -1
        for result in results:
            if result.keypoints is not None:
                for keypoint in result.keypoints:
                    if keypoint is not None:
                        keypoints_extracted = get_keypoints(keypoint)
                        if keypoints_extracted[2][1] > max_leg:
                            max_leg = keypoints_extracted[2][1]
                            points_for_frame = keypoints_extracted
        points_by_frame.append(points_for_frame)
        print(f"Frame {len(points_by_frame)}: {points_for_frame}")

    # get the difference between the current frame and the previous frame for the right wrist and plot it
    diff = [0]
    lankle_y = [0]
    rankle_y = [0]
    #smoothen each joint x and y position
    points_by_frame = np.array(points_by_frame)
    print(points_by_frame.shape)
    points_by_frame = np.nan_to_num(points_by_frame, nan=0.0, posinf=0.0, neginf=0.0)
    #for each frame, if val is 0.0, replace it with the previous value
    for i in range(1, len(points_by_frame)):
        for j in range(3):  # Assuming 3 keypoints: right wrist, right ankle, left ankle
            if points_by_frame[i][j][0] == 0.0:
                points_by_frame[i][j] = points_by_frame[i-1][j]
    points_by_frame = points_by_frame[:, :, :2]  # Keep only x and y coordinates
    points_by_frame = np.apply_along_axis(lambda x: np.convolve(x, np.ones(5)/5, mode='valid'), axis=0, arr=points_by_frame)
    print(points_by_frame.shape)
    #points_by_frame = points_by_frame.tolist()  # Convert back to list for iteration
    for i in range(1, len(points_by_frame)):
        right_wrist_current = points_by_frame[i][0]
        right_wrist_previous = points_by_frame[i-1][0]
        left_ankle_current = points_by_frame[i][1]
        left_ankle_previous = points_by_frame[i-1][1]
        right_ankle_current = points_by_frame[i][2]
        right_ankle_previous = points_by_frame[i-1][2]
        if left_ankle_current[0] > 0.0 and left_ankle_previous[0] > 0.0:
            difference = np.linalg.norm(left_ankle_current[:2] - left_ankle_previous[:2])
            print(f"Difference in left ankle position between frame {i} and {i-1}: {difference}")
            lankle_y.append(left_ankle_current[1])
        else:
            lankle_y.append(lankle_y[-1])
        if right_ankle_current[0] > 0.0 and right_ankle_previous[0] > 0.0:
            difference = np.linalg.norm(right_ankle_current[:2] - right_ankle_previous[:2])
            print(f"Difference in right ankle position between frame {i} and {i-1}: {difference}")
            rankle_y.append(right_ankle_current[1])
        else:
            rankle_y.append(rankle_y[-1])
        if right_wrist_current[0] > 0.0 and right_wrist_previous[0] > 0.0:
            difference = np.linalg.norm(right_wrist_current[:2] - right_wrist_previous[:2])
            print(f"Difference in right wrist position between frame {i} and {i-1}: {difference}")
            diff.append(difference)
        else:
            diff.append(diff[-1])
    #smooth diff, lankle_y, and rankle_y
    diff = np.convolve(diff[1:], np.ones(10)/10, mode='valid')
    # lankle_y = np.convolve(lankle_y, np.ones(5)/5, mode='valid')
    # rankle_y = np.convolve(rankle_y, np.ones(5)/5, mode='valid')
    #normalize the diff, lankle_y, and rankle_y
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    #cap lankle_y and diff to range of 0 to 10
    lankle_y = np.array(lankle_y)
    rankle_y = np.array(rankle_y)
    # lankle_y = np.clip(lankle_y, 0, 10)
    # diff = np.clip(diff, 0, 10)
    lankle_y = lankle_y[1:]  # Remove the first element to match the length of diff
    rankle_y = rankle_y[1:]  # Remove the first element to match the length of diff 
    lankle_y = (lankle_y - np.min(lankle_y)) / (np.max(lankle_y) - np.min(lankle_y))
    rankle_y = (rankle_y - np.min(rankle_y)) / (np.max(rankle_y) - np.min(rankle_y))
    lankle_y_diff = np.diff(lankle_y, prepend=lankle_y[0])  # Prepend the first value to maintain the same length
    rankle_y_diff = np.diff(rankle_y, prepend=rankle_y[0])  # Prepend the first value to maintain the same length
    #smooth the differences
    lankle_y_diff = np.convolve(lankle_y_diff, np.ones(10)/10, mode='valid')
    rankle_y_diff = np.convolve(rankle_y_diff, np.ones(10)/10, mode='valid')
    lankle_y_diff = np.convolve(lankle_y_diff, np.ones(5)/5, mode='valid')
    rankle_y_diff = np.convolve(rankle_y_diff, np.ones(5)/5, mode='valid')

    lankle_y_diff = np.where(lankle_y_diff < 0.01, lankle_y_diff, .75)
    rankle_y_diff = np.where(rankle_y_diff < 0.02, rankle_y_diff, 1)
    lankle_y_diff = np.where(lankle_y_diff > -0.01, lankle_y_diff, -.75)
    rankle_y_diff = np.where(rankle_y_diff > -0.02, rankle_y_diff, -1)
    #identify all frames where it goes from less than -0.5 to positive 0.5 within 10 frames
    lankle_change = []
    for i in range(len(lankle_y_diff) - 20):
        if lankle_y_diff[i] < -0.5 and np.any(lankle_y_diff[i:i+20] > 0.5):
            #if the prev value in lankle_change is within 5 frames, replace it
            if lankle_change and i - lankle_change[-1] < 5:
                lankle_change[-1] = i
            else:
                lankle_change.append(i)
    rankle_change = []
    for i in range(len(rankle_y_diff) - 20):
        if rankle_y_diff[i] < -0.5 and np.any(rankle_y_diff[i:i+20] > 0.5):
            #if the prev value in lankle_change is within 5 frames, replace it
            if rankle_change and i - rankle_change[-1] < 5:
                rankle_change[-1] = i
            else:
                rankle_change.append(i)
    print(f"Left Ankle Recovery Points: {lankle_change}")
    print(f"Right Ankle Recovery Points: {rankle_change}")

    # Go through each value in rankle_change, for each rankle_change, check if there is a lankle_change 40 to 95 frames before it, if so, print the lankle_change and the rankle_change
    recovery = ()
    for rankle in rankle_change:
        for lankle in lankle_change:
            if lankle < rankle and rankle - lankle > 22 and rankle - lankle < 95 and recovery == ():
                recovery = (lankle, rankle)
                #break from both for loops
                break
    if recovery != ():
        #Go through lankle_y_diff from recovery[0] to the beginning
        for i in range(recovery[0], -1, -1):
            if lankle_y_diff[i] < -0.5 and recovery[0] - i < 10:
                recovery = (i, recovery[1])
        # Go through rankle_y_diff from recovery[1] to the end
        for i in range(recovery[1]+10, len(rankle_y_diff)):
            if rankle_y_diff[i] > 0.5 and i - recovery[1] < 20:
                recovery = (recovery[0], i)
        recovery = (max(recovery[0]-5, 0), min(recovery[1]+20, len(rankle_y_diff)))



    #plt.plot(diff, color='red', label='Right Wrist Position Difference')
    x = np.arange(len(rankle_y_diff))
    plt.plot(lankle_y, label='Left Ankle Y Position')
    plt.plot(rankle_y, label='Right Ankle Y Position', color='orange')
    plt.scatter(x,lankle_y_diff, label='Left Ankle Y Position Difference', color='blue')
    plt.scatter(x,rankle_y_diff, label='Right Ankle Y Position Difference', color='purple')
    plt.legend()
    plt.grid()
    plt.title('Difference in Right Ankle Position Over Frames')
    plt.xlabel('Frame Index')
    plt.ylabel('Difference (pixels)')
    plt.show()
    plt.savefig("pose_analysis_plot.png")




    cap.release()
    cv2.destroyAllWindows()
            #return recovery if it is not empty
    if recovery == ():
        print("No recovery points found.")
        return None
    else:
        return recovery



# Example usage:
# pose_and_overlay_video('input.mp4')

def clean_single_mp4_video(video_path):
    """
    Process a single mp4 video: detect recovery points, crop the video between those points,
    and overwrite the original video with the cropped version.
    """
    print(f"Processing: {video_path}")

    try:
        recovery = pose_and_overlay_video(video_path)
        if recovery is not None:
            start_frame, end_frame = recovery
            print(f"Recovery points found in {video_path}: start={start_frame}, end={end_frame}")

            # Load original video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Write to a temporary file first
            temp_path = video_path + ".temp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_idx > end_frame:
                    break
                if start_frame <= frame_idx <= end_frame:
                    out.write(frame)
                frame_idx += 1

            cap.release()
            out.release()

            # Overwrite original video
            import os
            os.replace(temp_path, video_path)
            print(f"Overwritten original video with cropped segment at {video_path}")
            return True
        else:
            print(f"No recovery points found in {video_path}.")

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
    return False



def flip_video_frames(frames_array):
    """
    Flips the video frames horizontally.

    Args:
        frames_array (np.ndarray): Input video as a numpy array (frames, H, W, C). or (H, W, C) for a single image.

    Returns:
        np.ndarray: Flipped video as a numpy array.
    """
    # If input is a single image (3D array), flip along width axis
    if isinstance(frames_array, np.ndarray):
        if frames_array.ndim == 3:
            return np.flip(frames_array, axis=1)
        elif frames_array.ndim == 4:
            return np.flip(frames_array, axis=2)
    raise ValueError("Input must be a numpy array with 3 (image) or 4 (video) dimensions.")



if __name__ == "__main__":
    # Example usage
    # Replace with the path to your video file
    # clean_single_mp4_video('path_to_your_video.mp4')
    # For testing, you can use a sample video file
    # clean_single_mp4_video('/home/ec2-user/shadow-trainer/api_backend/tmp_api_output/Left_Hand_Friend_Front.MOV')
    clean_single_mp4_video('/home/ec2-user/shadow-trainer/api_backend/tmp_api_output/Left_Hand_Friend_Diagonal.MOV')