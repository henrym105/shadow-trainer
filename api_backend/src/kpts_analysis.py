
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import openai
import numpy as np
import math
import os
from dotenv import load_dotenv
# from constants import API_ROOT_DIR

API_ROOT_DIR = Path(__file__).parent.parent
print(API_ROOT_DIR)

# Joint names (unchanged)
JOINT_NAMES = {
    0: 'Pelvis',
    1: 'Left Hip',
    2: 'Right Hip',
    3: 'Left Knee',
    4: 'Right Knee',
    5: 'Left Ankle',
    6: 'Right Ankle',
    7: 'Spine',
    8: 'Thorax',
    9: 'Neck',
    10: 'Head',
    11: 'Left Shoulder',
    12: 'Right Shoulder',
    13: 'Left Elbow',
    14: 'Right Elbow',
    15: 'Left Wrist',
    16: 'Right Wrist',
}



LLM_PROMPT = """
Summarize this information for someone who wants quick insights.
This is a comparison between the movement of a user and a professional performing the same motion.
Respond with just a few bullet points that offer clear, actionable suggestions the user can follow to better match the professional's form.
Avoid repeating metrics or scores — focus on real-world movement insights.
Do not include emojis. Keep the language simple and practical.
Here is the data:
{}
Limit your response to 3 bullet points maximum.

--- MOTION FEEDBACK RUBRIC & INTERPRETATION GUIDE ---
JOINT DISTANCE (Euclidean Distance):
- Measures how far the user's joint is from the professional's (in meters) per frame.
- Lower values = better mimicry of joint position.
Interpretation:
- < 0.05 m     : Excellent match
- 0.05 - 0.10 m  : Good
- 0.10 - 0.20 m  : Fair
- > 0.20 m     : Poor (needs work)
JOINT ANGLE DIFFERENCE (degrees):
- Measures how much the joint's bending/rotation angle differs from pro.
- Lower values = better mimicry of joint movement.
Interpretation:
- < 5°         : Excellent
- 5° - 10°       : Good
- 10° - 20°      : Fair
- > 20°        : Poor (likely off-form)
HIP TORSION SPEED (deg/s):
- Measures rotational speed of hip (left vs right) around vertical axis.
Key Metrics:
- MAE (Mean Absolute Error): < 10°/s Excellent, > 50°/s Poor
- Similarity Score (0 - 1): > 0.9 Excellent, < 0.5 Poor
- Peak Speed Diff: Large mismatch may indicate rotational inefficiency
- Timing Diff: -ve = user peaks earlier; +ve = later
- Speed Correlation: > 0.8 Excellent, < 0.3 Poor
- Std Dev: Low = consistent movement, High = variability
IMPROVEMENT TIPS:
- High distance or angle error: slow down and focus on joint positioning
- Poor hip torsion: work on core/hip flexibility and timing
- Inconsistent speed: aim for smoother motion with better control
"""



def angle_between_points(a, b, c):
    """
    Calculate the angle (in degrees) at point b given points a, b, c (each np.array shape (3,))
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return np.nan
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Numerical safety
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

def compute_joint_angle_over_time(kps, joint_idx):
    """
    Compute joint angles over time for selected joints based on triplets (parent, joint, child).
    Returns a numpy array of angles (degrees) per frame, or NaNs if undefined.
    """

    # Define triplets (parent, joint, child) for angle calculations
    joint_angle_map = {
        # Elbows
        13: (11, 13, 15),  # Left Elbow: Shoulder-Elbow-Wrist
        14: (12, 14, 16),  # Right Elbow

        # Shoulders (approximate, using Spine (7) as parent)
        11: (7, 11, 13),   # Left Shoulder: Spine-Shoulder-Elbow
        12: (7, 12, 14),   # Right Shoulder

        # Hips
        1: (0, 1, 3),      # Left Hip: Pelvis-Hip-Knee
        2: (0, 2, 4),      # Right Hip

        # Knees
        3: (1, 3, 5),      # Left Knee: Hip-Knee-Ankle
        4: (2, 4, 6),      # Right Knee
    }

    if joint_idx not in joint_angle_map:
        # Angle calculation not defined for this joint
        return np.full(kps.shape[0], np.nan)

    parent_idx, joint_idx_inner, child_idx = joint_angle_map[joint_idx]
    angles = []
    for frame in range(kps.shape[0]):
        a = kps[frame, parent_idx, :]
        b = kps[frame, joint_idx_inner, :]
        c = kps[frame, child_idx, :]

        # If any keypoint is missing (NaN), skip angle calc
        if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
            angles.append(np.nan)
            continue

        angle = angle_between_points(a, b, c)
        angles.append(angle)

    return np.array(angles)

def joint_similarity_score(user_kps, pro_kps, joint_idx, threshold=0.5, angle_threshold=30, plot=False):
    if joint_idx == -1:
        # Hip torsion speed calculation
        def compute_hip_torsion_speed(kps, fps=30):
            left_hip = kps[:, 1, :2]
            right_hip = kps[:, 2, :2]
            hip_vec = right_hip - left_hip
            hip_angles = np.arctan2(hip_vec[:, 1], hip_vec[:, 0])
            hip_angles_unwrapped = np.unwrap(hip_angles)
            angular_velocity_rad = np.diff(hip_angles_unwrapped) * fps
            torsion_speeds = np.rad2deg(np.abs(angular_velocity_rad))
            return torsion_speeds, np.rad2deg(hip_angles_unwrapped)

        fps = 30
        min_frames = min(user_kps.shape[0], pro_kps.shape[0])
        user_speed, _ = compute_hip_torsion_speed(user_kps[:min_frames], fps)
        pro_speed, _ = compute_hip_torsion_speed(pro_kps[:min_frames], fps)

        min_len = min(len(user_speed), len(pro_speed))
        user_speed = user_speed[:min_len]
        pro_speed = pro_speed[:min_len]

        mae = np.mean(np.abs(user_speed - pro_speed))
        torsion_score = np.clip(1 - (mae / 100), 0, 1)

        peak_user = np.max(user_speed)
        peak_pro = np.max(pro_speed)

        avg_user = np.mean(user_speed)
        avg_pro = np.mean(pro_speed)

        timing_diff = np.argmax(user_speed) - np.argmax(pro_speed)

        corr = np.corrcoef(user_speed, pro_speed)[0,1]

        std_user = np.std(user_speed)

        if plot:
            plt.figure(figsize=(10,4))
            plt.plot(user_speed, label='User Hip Torsion Speed')
            plt.plot(pro_speed, label='Pro Hip Torsion Speed')
            plt.xlabel('Frame')
            plt.ylabel('Torsion Speed (deg/s)')
            plt.title('Hip Torsion Speed Comparison')
            plt.legend()
            plt.grid(True)
            plt.show()

            print("Hip Torsion Speed Feedback:")
            print(f"- Mean Absolute Error (MAE): {mae:.3f} deg/s")
            print(f"- Similarity Score: {torsion_score:.3f} (1 = perfect match)")
            print(f"- Peak Speed User / Pro: {peak_user:.2f} / {peak_pro:.2f} deg/s")
            print(f"- Average Speed User / Pro: {avg_user:.2f} / {avg_pro:.2f} deg/s")
            print(f"- Timing Difference (peak frame offset): {timing_diff} frames")
            print(f"- Speed Curve Correlation: {corr:.3f}")
            print(f"- User Speed Consistency (std dev): {std_user:.2f}")
            print()
            print("Interpretation:")
            print("A lower MAE and higher similarity score indicate your hip rotation speed closely matches the professional's.")
            print("Peak speeds show the max rotational velocity; big differences may suggest different mechanics.")
            print("Timing difference shows if your hip rotation peaks earlier or later.")
            print("Correlation measures how similarly your rotation speed varies frame-to-frame.")
            print("A high std dev in your speed may mean inconsistent hip rotation.")

        return mae, torsion_score, np.nan, np.nan

    # Regular joint processing
    joint_name = JOINT_NAMES.get(joint_idx, f'Joint {joint_idx}')
    min_frames = min(user_kps.shape[0], pro_kps.shape[0])

    user_joint = user_kps[:min_frames, joint_idx, :]
    pro_joint = pro_kps[:min_frames, joint_idx, :]

    distances = np.linalg.norm(user_joint - pro_joint, axis=1)
    avg_distance = distances.mean()
    max_distance = distances.max()
    min_distance = distances.min()
    std_distance = distances.std()
    dist_score = np.clip(1 - (avg_distance / threshold), 0, 1)

    user_angles = compute_joint_angle_over_time(user_kps[:min_frames], joint_idx)
    pro_angles = compute_joint_angle_over_time(pro_kps[:min_frames], joint_idx)

    if np.isnan(user_angles).all() or np.isnan(pro_angles).all():
        avg_angle_diff = np.nan
        angle_score = np.nan
    else:
        angle_diffs = np.abs(user_angles - pro_angles)
        avg_angle_diff = np.nanmean(angle_diffs)
        angle_score = np.clip(1 - (avg_angle_diff / angle_threshold), 0, 1)

    if plot:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(distances, label='Euclidean Distance')
        plt.xlabel('Frame')
        plt.ylabel('Distance (m)')
        plt.title(f'{joint_name} - Distance Over Time')
        plt.legend()
        plt.grid(True)

        plt.subplot(1,2,2)
        if not np.isnan(avg_angle_diff):
            plt.plot(user_angles, label='User Angle')
            plt.plot(pro_angles, label='Pro Angle')
            plt.xlabel('Frame')
            plt.ylabel('Angle (degrees)')
            plt.title(f'{joint_name} - Angle Over Time')
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Angle data not available for this joint',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Detailed textual feedback:
        print(f"Tracking {joint_name}:")
        print(f"Distance - Mean: {avg_distance:.4f}, Max: {max_distance:.4f}, Min: {min_distance:.4f}, Std: {std_distance:.4f}")
        print(f"Distance Similarity Score: {dist_score:.3f} (1 = perfect match)")
        print("Interpretation:")
        print(f"- The mean distance indicates how far your {joint_name.lower()} joint's position is from the pro's on average (in meters).")
        print(f"- Lower distances mean your joint positions closely follow the professional's motion.")
        print(f"- The similarity score (0-1) normalizes this distance: 1 means very close match, 0 means very different.")
        if not np.isnan(avg_angle_diff):
            print(f"Angle - Mean Difference: {avg_angle_diff:.2f} degrees")
            print(f"Angle Similarity Score: {angle_score:.3f} (1 = perfect match)")
            print(f"- The mean angle difference shows how closely your joint bending or rotation matches the professional's.")
            print(f"- Lower angle differences and higher similarity scores indicate better mimicry of the joint movement.")
        else:
            print("- Angle data not available for this joint.")

    return avg_distance, dist_score, avg_angle_diff, angle_score


def evaluate_all_joints(user_kps, pro_kps, plot=False) -> dict:
    EXCLUDED_JOINTS = [0, 7, 8, 9, 10]  # Pelvis, Spine, Thorax, Neck, Head
    joints_to_eval = [j for j in range(user_kps.shape[1]) if j not in EXCLUDED_JOINTS]

    results = {}

    # Hip torsion special case
    mae, score, _, _ = joint_similarity_score(user_kps, pro_kps, joint_idx=-1, plot=plot)
    results['Hip Torsion'] = {'MAE': mae, 'Similarity Score': score}

    for j in joints_to_eval:
        avg_dist, dist_score, avg_angle_diff, angle_score = joint_similarity_score(
            user_kps, pro_kps, joint_idx=j, plot=plot)
        joint_name = JOINT_NAMES.get(j, f'Joint {j}')
        results[joint_name] = {
            'Avg Distance': avg_dist,
            'Distance Score': dist_score,
            'Avg Angle Diff': avg_angle_diff,
            'Angle Score': angle_score,
        }

    return results


def evaluate_all_joints_text(user_kps, pro_kps) -> str:
    """Generate a text summary of joint evaluation results"""
    EXCLUDED_JOINTS = [0, 7, 8, 9, 10]  # Pelvis, Spine, Thorax, Neck, Head
    joints_to_eval = [j for j in range(user_kps.shape[1]) if j not in EXCLUDED_JOINTS]

    output = []
    output.append("Joint Movement Analysis Summary")
    output.append("=" * 35)
    output.append("")

    # Hip torsion special case
    mae, score, _, _ = joint_similarity_score(user_kps, pro_kps, joint_idx=-1, plot=False)
    output.append("Hip Torsion:")
    if not np.isnan(mae):
        output.append(f"  • Mean Absolute Error: {mae:.3f} deg/s")
    else:
        output.append("  • Mean Absolute Error: N/A")
    
    if not np.isnan(score):
        output.append(f"  • Similarity Score: {score:.3f}")
    else:
        output.append("  • Similarity Score: N/A")
    output.append("")

    for j in joints_to_eval:
        avg_dist, dist_score, avg_angle_diff, angle_score = joint_similarity_score(
            user_kps, pro_kps, joint_idx=j, plot=False)
        joint_name = JOINT_NAMES.get(j, f'Joint {j}')
        
        output.append(f"{joint_name}:")
        
        # Distance metrics
        if not np.isnan(avg_dist):
            output.append(f"  • Average Distance: {avg_dist:.4f}m")
        else:
            output.append("  • Average Distance: N/A")
            
        if not np.isnan(dist_score):
            output.append(f"  • Distance Score: {dist_score:.3f}")
        else:
            output.append("  • Distance Score: N/A")
            
        # Angle metrics
        if not np.isnan(avg_angle_diff):
            output.append(f"  • Average Angle Difference: {avg_angle_diff:.2f}°")
        else:
            output.append("  • Average Angle Difference: N/A")
            
        if not np.isnan(angle_score):
            output.append(f"  • Angle Score: {angle_score:.3f}")
        else:
            output.append("  • Angle Score: N/A")
            
        output.append("")

    output.append("Interpretation:")
    output.append("• Distance scores show how closely your joint positions match the professional's")
    output.append("• Angle scores indicate how well your joint movements replicate the professional's technique")
    output.append("• Higher scores (closer to 1.0) indicate better similarity to the professional athlete")

    return "\n".join(output)


def summarize_joint_results(joint_text) -> str:
    """
    Generates a prompt summarizing joint_text in plain English without metric repetition.
    """
    prompt = LLM_PROMPT.format(joint_text)
    return prompt


def generate_motion_feedback(joint_text) -> str:
    API_ROOT_DIR = Path(__file__).parent.parent
    dotenv_path = os.path.join(API_ROOT_DIR, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = summarize_joint_results(joint_text)
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4-mini" if available
        messages=[
            {"role": "system", "content": "You are a biomechanics coach giving feedback based on motion analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Example usage
    pro_kps = np.load("/home/ec2-user/shadow-trainer/api_backend/sample_videos/sample_output/raw_keypoints/pro_3D_keypoints.npy")
    user_kps = np.load("/home/ec2-user/shadow-trainer/api_backend/sample_videos/sample_output/raw_keypoints/user_3D_keypoints.npy")

    print("User shape:", user_kps.shape)
    print("Pro shape:", pro_kps.shape)

    # Evaluate all joints and print results
    # results = evaluate_all_joints(user_kps, pro_kps, plot=True)
    # for joint, metrics in results.items():
    #     print(f"{joint}: {metrics}")

    results = evaluate_all_joints_text(user_kps, pro_kps)
    print(results)

    print("-"*100)
    feedback = generate_motion_feedback(results)
    print(feedback)




    # """
    # hip direction angle difference: [0,0,1,1,2,2,2,3,3,4,4,3,2,1,0, -1, -1, -1, -2, -2, -3, -3, -4, -4, -3, -2, -1, 0]
    # - positive numbers indicate the user has rotating their hips more than the pro at that point of the motion
    # - negative numbers indicate the user has rotated their hips less than the pro
    # """
