import numpy as np
import matplotlib.pyplot as plt
from math import acos, degrees

# Joints to exclude from analysis
EXCLUDE_JOINTS = [0, 7, 8, 9, 10]  # Pelvis, Spine, Thorax, Neck, Head

# Joint names and groups
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

JOINT_GROUPS = {
    'Left Arm': [11, 13, 15],
    'Right Arm': [12, 14, 16],
    'Left Leg': [1, 3, 5],
    'Right Leg': [2, 4, 6],
    'Shoulders': [11, 12],
    'Hips': [1, 2],
}

ANGLE_TRIPLETS = {
    13: (11, 13, 15),
    14: (12, 14, 16),
    3: (1, 3, 5),
    4: (2, 4, 6),
    11: (7, 11, 13),
    12: (7, 12, 14),
}

def vector_angle(v1, v2):
    dot = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = np.clip(dot / (norm_product + 1e-8), -1.0, 1.0)
    return degrees(acos(cos_angle))

def compute_joint_angle(keypoints, triplet):
    a, b, c = triplet
    vec1 = keypoints[:, a] - keypoints[:, b]
    vec2 = keypoints[:, c] - keypoints[:, b]
    return np.array([vector_angle(v1, v2) for v1, v2 in zip(vec1, vec2)])

def angular_velocity(angles, dt=1.0):
    return np.gradient(angles, dt)

def euclidean_distance_over_time(arr1, arr2):
    return np.linalg.norm(arr1 - arr2, axis=1)

def fmt_val(val, digits=3):
    if val is None:
        return "N/A"
    try:
        return f"{val:.{digits}f}"
    except:
        return str(val)

def compute_abduction_angle(keypoints, shoulder=12, elbow=14, spine=7):
    vec1 = keypoints[:, shoulder] - keypoints[:, spine]
    vec2 = keypoints[:, elbow] - keypoints[:, shoulder]
    return np.array([vector_angle(v1, v2) for v1, v2 in zip(vec1, vec2)])

def compute_torsion_angle(keypoints, left_joint, right_joint):
    center = (keypoints[:, left_joint] + keypoints[:, right_joint]) / 2
    left_vec = keypoints[:, left_joint] - center
    right_vec = keypoints[:, right_joint] - center
    return np.array([vector_angle(lv, rv) for lv, rv in zip(left_vec, right_vec)])

def compute_rotational_speed_arm(keypoints, joint, fps=30.0):
    """Compute rotational speed for arm joints relative to thorax in degrees per second"""
    # Use thorax (joint 8) as reference point for arms
    thorax = keypoints[:, 8]
    
    # Calculate position vectors from thorax to joint
    position_vectors = keypoints[:, joint] - thorax
    
    # Calculate frame-to-frame rotational speeds
    rotational_speeds = []
    
    for i in range(1, len(position_vectors)):
        # Previous and current position vectors
        prev_vec = position_vectors[i-1]
        curr_vec = position_vectors[i]
        
        # Calculate angle between consecutive vectors (using 2D projection for rotation around vertical axis)
        prev_vec_2d = prev_vec[:2]  # x, y components
        curr_vec_2d = curr_vec[:2]
        
        # Calculate angle between vectors in radians
        dot_product = np.dot(prev_vec_2d, curr_vec_2d)
        cross_product = np.cross(prev_vec_2d, curr_vec_2d)
        angle_rad = np.arctan2(cross_product, dot_product)
        
        # Convert to degrees and then to degrees per second
        angle_deg = np.degrees(angle_rad)
        speed_deg_per_sec = angle_deg * fps
        
        rotational_speeds.append(speed_deg_per_sec)
    
    # Add zero for first frame (no previous frame to compare)
    return np.array([0.0] + rotational_speeds)

def compute_rotational_speed_leg(keypoints, joint, fps=30.0):
    """Compute rotational speed for leg joints relative to pelvis in degrees per second"""
    # Use pelvis (joint 0) as reference point for legs
    pelvis = keypoints[:, 0]
    
    # Calculate position vectors from pelvis to joint
    position_vectors = keypoints[:, joint] - pelvis
    
    # Calculate frame-to-frame rotational speeds
    rotational_speeds = []
    
    for i in range(1, len(position_vectors)):
        # Previous and current position vectors
        prev_vec = position_vectors[i-1]
        curr_vec = position_vectors[i]
        
        # Calculate angle between consecutive vectors (using 2D projection for rotation around vertical axis)
        prev_vec_2d = prev_vec[:2]  # x, y components
        curr_vec_2d = curr_vec[:2]
        
        # Calculate angle between vectors in radians
        dot_product = np.dot(prev_vec_2d, curr_vec_2d)
        cross_product = np.cross(prev_vec_2d, curr_vec_2d)
        angle_rad = np.arctan2(cross_product, dot_product)
        
        # Convert to degrees and then to degrees per second
        angle_deg = np.degrees(angle_rad)
        speed_deg_per_sec = angle_deg * fps
        
        rotational_speeds.append(speed_deg_per_sec)
    
    # Add zero for first frame (no previous frame to compare)
    return np.array([0.0] + rotational_speeds)

def extended_evaluation(user_kps, pro_kps, plot=True):
    T, J, _ = user_kps.shape
    assert pro_kps.shape == (T, J, 3), "User and pro keypoints must have same shape"

    joint_metrics = {}

    # Compute per joint metrics
    for j in range(J):
        if j in EXCLUDE_JOINTS:
            continue

        dist_t = euclidean_distance_over_time(user_kps[:, j], pro_kps[:, j])
        dist_mae = np.mean(dist_t) if len(dist_t) > 0 else None

        if j in ANGLE_TRIPLETS:
            user_angles = compute_joint_angle(user_kps, ANGLE_TRIPLETS[j])
            pro_angles = compute_joint_angle(pro_kps, ANGLE_TRIPLETS[j])
            angle_diff = np.abs(user_angles - pro_angles)
            angle_mae = np.mean(angle_diff)

            # Use rotational speed for arm and leg joints
            if j in [11, 12, 13, 14, 15, 16]:  # Arm joints
                user_ang_vel = compute_rotational_speed_arm(user_kps, j)
                pro_ang_vel = compute_rotational_speed_arm(pro_kps, j)
            elif j in [1, 2, 3, 4, 5, 6]:  # Leg joints
                user_ang_vel = compute_rotational_speed_leg(user_kps, j)
                pro_ang_vel = compute_rotational_speed_leg(pro_kps, j)
            else:
                user_ang_vel = angular_velocity(user_angles)
                pro_ang_vel = angular_velocity(pro_angles)
            
            ang_vel_diff = np.abs(user_ang_vel - pro_ang_vel)
            ang_vel_mae = np.mean(ang_vel_diff)
        else:
            user_angles = None
            pro_angles = None
            angle_mae = None
            user_ang_vel = None
            pro_ang_vel = None
            ang_vel_mae = None

        joint_metrics[j] = {
            'name': JOINT_NAMES.get(j, f'Joint_{j}'),
            'dist_t': dist_t,
            'dist_mae': dist_mae,
            'user_angles': user_angles,
            'pro_angles': pro_angles,
            'angle_mae': angle_mae,
            'user_ang_vel': user_ang_vel,
            'pro_ang_vel': pro_ang_vel,
            'ang_vel_mae': ang_vel_mae,
        }

    # Compute group mean values (MAEs)
    group_metrics = {}
    for group_name, joint_ids in JOINT_GROUPS.items():
        filtered_ids = [j for j in joint_ids if j in joint_metrics]
        dist_maes = [joint_metrics[j]['dist_mae'] for j in filtered_ids if joint_metrics[j]['dist_mae'] is not None]
        angle_maes = [joint_metrics[j]['angle_mae'] for j in filtered_ids if joint_metrics[j]['angle_mae'] is not None]
        ang_vel_maes = [joint_metrics[j]['ang_vel_mae'] for j in filtered_ids if joint_metrics[j]['ang_vel_mae'] is not None]

        group_metrics[group_name] = {
            'dist_mae': np.mean(dist_maes) if dist_maes else None,
            'angle_mae': np.mean(angle_maes) if angle_maes else None,
            'ang_vel_mae': np.mean(ang_vel_maes) if ang_vel_maes else None,
            'joint_ids': filtered_ids
        }

    # Biomechanical features for spider chart & overall score
    right_arm_joints = JOINT_GROUPS['Right Arm']  # [12, 14, 16]

    throwing_arm_user_angles = []
    throwing_arm_pro_angles = []

    for joint in right_arm_joints:
        if joint in ANGLE_TRIPLETS:
            # Use rotational speed for throwing arm analysis
            throwing_arm_user_angles.append(compute_rotational_speed_arm(user_kps, joint))
            throwing_arm_pro_angles.append(compute_rotational_speed_arm(pro_kps, joint))

    if throwing_arm_user_angles and throwing_arm_pro_angles:
        throwing_arm_user_angles = np.array(throwing_arm_user_angles)
        throwing_arm_pro_angles = np.array(throwing_arm_pro_angles)

        throwing_arm_user_mean_angle = np.mean(throwing_arm_user_angles, axis=0)
        throwing_arm_pro_mean_angle = np.mean(throwing_arm_pro_angles, axis=0)

        throwing_arm_angle_mae = np.mean(np.abs(throwing_arm_user_mean_angle - throwing_arm_pro_mean_angle))
    else:
        throwing_arm_user_mean_angle = None
        throwing_arm_pro_mean_angle = None
        throwing_arm_angle_mae = None

    user_abduction = compute_abduction_angle(user_kps)
    pro_abduction = compute_abduction_angle(pro_kps)
    abduction_mae = np.mean(np.abs(user_abduction - pro_abduction))

    user_hip_torsion = compute_torsion_angle(user_kps, 1, 2)
    pro_hip_torsion = compute_torsion_angle(pro_kps, 1, 2)
    hip_torsion_mae = np.mean(np.abs(user_hip_torsion - pro_hip_torsion))

    user_shoulder_torsion = compute_torsion_angle(user_kps, 11, 12)
    pro_shoulder_torsion = compute_torsion_angle(pro_kps, 11, 12)
    shoulder_torsion_mae = np.mean(np.abs(user_shoulder_torsion - pro_shoulder_torsion))

    # Normalized similarity score function
    def normalized_score(mae, max_val):
        if mae is None:
            return None
        return max(0, 1 - mae / max_val)

    biomech_maes = {
        'Throwing Arm Angular Velocity': throwing_arm_angle_mae,
        'Right Shoulder Abduction': abduction_mae,
        'Hip Torsion': hip_torsion_mae,
        'Shoulder Torsion': shoulder_torsion_mae,
    }

    biomech_scores = {k: normalized_score(v, max_val=30) if v is not None else None for k, v in biomech_maes.items()}
    overall_score = np.mean([v for v in biomech_scores.values() if v is not None])

    # Print raw MAE values for joints
    print("\nJoint Metrics (Mean Absolute Error):")
    for j, m in joint_metrics.items():
        print(f"  {m['name']:15s} | Dist MAE: {fmt_val(m['dist_mae'])} meters, Angle MAE: {fmt_val(m['angle_mae'])}°, Rotational Speed MAE: {fmt_val(m['ang_vel_mae'])} °/s")

    # Print group mean MAEs
    print("\nGroup Metrics (Mean Absolute Error):")
    for group, m in group_metrics.items():
        print(f"  {group:10s} | Dist MAE: {fmt_val(m['dist_mae'])} meters, Angle MAE: {fmt_val(m['angle_mae'])}°, Rotational Speed MAE: {fmt_val(m['ang_vel_mae'])} °/s")

    # Print biomechanical feature MAEs and scores
    print("\nBiomechanical Features (MAE and similarity score):")
    for k in biomech_maes.keys():
        print(f"  {k:30s} | MAE: {fmt_val(biomech_maes[k])} | Similarity Score: {fmt_val(biomech_scores[k])}")

    print(f"\nOverall Biomechanical Similarity Score: {fmt_val(overall_score * 100)}%")

    if plot:
        # Per-joint plots
        for j, m in joint_metrics.items():
            fig, axs = plt.subplots(1, 3, figsize=(18, 4))
            fig.suptitle(f"Joint: {m['name']}")

            # Distance
            axs[0].plot(m['dist_t'], label='Distance')
            axs[0].set_ylabel('Euclidean Distance (meters)')
            axs[0].set_xlabel('Frame')
            axs[0].grid(True)

            # Angle
            if m['user_angles'] is not None and m['pro_angles'] is not None:
                axs[1].plot(m['user_angles'], label='User Angle (°)')
                axs[1].plot(m['pro_angles'], label='Pro Angle (°)', alpha=0.7)
            axs[1].set_ylabel('Angle (degrees)')
            axs[1].set_xlabel('Frame')
            axs[1].grid(True)

            # Rotational speed
            if m['user_ang_vel'] is not None and m['pro_ang_vel'] is not None:
                axs[2].plot(m['user_ang_vel'], label='User Rotational Speed (°/s)')
                axs[2].plot(m['pro_ang_vel'], label='Pro Rotational Speed (°/s)', alpha=0.7)
            axs[2].set_ylabel('Rotational Speed (degrees/second)')
            axs[2].set_xlabel('Frame')
            axs[2].grid(True)

            for ax in axs:
                ax.legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"joint_{j:02d}_{m['name'].replace(' ', '_').lower()}_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Group mean curves
        for group_name, group_data in group_metrics.items():
            joint_ids = group_data['joint_ids']
            if not joint_ids:
                continue

            dist_mat = np.array([joint_metrics[j]['dist_t'] for j in joint_ids if joint_metrics[j]['dist_t'] is not None])
            dist_mean = dist_mat.mean(axis=0) if dist_mat.size > 0 else None

            angle_user_mat = []
            angle_pro_mat = []
            angle_user_mat = []
            angle_pro_mat = []

            for j in joint_ids:
                m = joint_metrics[j]
                if m['user_angles'] is not None and m['pro_angles'] is not None:
                    angle_user_mat.append(m['user_angles'])
                    angle_pro_mat.append(m['pro_angles'])
                if m['user_ang_vel'] is not None and m['pro_ang_vel'] is not None:
                    angle_user_mat.append(m['user_ang_vel'])
                    angle_pro_mat.append(m['pro_ang_vel'])

            angle_user_mean = np.mean(angle_user_mat, axis=0) if angle_user_mat else None
            angle_pro_mean = np.mean(angle_pro_mat, axis=0) if angle_pro_mat else None
            angle_user_mean = np.mean(angle_user_mat, axis=0) if angle_user_mat else None
            angle_pro_mean = np.mean(angle_pro_mat, axis=0) if angle_pro_mat else None

            fig, axs = plt.subplots(1, 3, figsize=(18, 4))
            fig.suptitle(f"Group: {group_name} Mean Curves")

            # Distance
            if dist_mean is not None:
                axs[0].plot(dist_mean, label='Mean Distance')
            axs[0].set_title("Euclidean Distance over Time")
            axs[0].set_ylabel("Distance (meters)")
            axs[0].set_xlabel("Frame")
            axs[0].grid(True)

            # Angle
            if angle_user_mean is not None and angle_pro_mean is not None:
                axs[1].plot(angle_user_mean, label='User Mean Angle (°)')
                axs[1].plot(angle_pro_mean, label='Pro Mean Angle (°)', alpha=0.7)
            axs[1].set_title("Angle over Time")
            axs[1].set_ylabel("Angle (degrees)")
            axs[1].set_xlabel("Frame")
            axs[1].grid(True)

            # Rotational speed
            if angle_user_mean is not None and angle_pro_mean is not None:
                axs[2].plot(angle_user_mean, label='User Mean Rotational Speed (°/s)')
                axs[2].plot(angle_pro_mean, label='Pro Mean Rotational Speed (°/s)', alpha=0.7)
            axs[2].set_title("Rotational Speed over Time")
            axs[2].set_ylabel("Rotational Speed (degrees/second)")
            axs[2].set_xlabel("Frame")
            axs[2].grid(True)

            for ax in axs:
                ax.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"group_{group_name.replace(' ', '_').lower()}_mean_curves.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Abduction angle plot
        plt.figure(figsize=(10, 4))
        plt.plot(user_abduction, label='User')
        plt.plot(pro_abduction, label='Pro', alpha=0.7)
        plt.title("Right Shoulder Abduction Angle Over Time")
        plt.ylabel("Angle (degrees)")
        plt.xlabel("Frame")
        plt.legend()
        plt.grid(True)
        plt.savefig("right_shoulder_abduction_angle.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Hip torsion plot
        plt.figure(figsize=(10, 4))
        plt.plot(user_hip_torsion, label='User Hip Torsion')
        plt.plot(pro_hip_torsion, label='Pro Hip Torsion', alpha=0.7)
        plt.title("Hip Torsion Angle Over Time")
        plt.ylabel("Angle (degrees)")
        plt.xlabel("Frame")
        plt.legend()
        plt.grid(True)
        plt.savefig("hip_torsion_angle.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Shoulder torsion plot
        plt.figure(figsize=(10, 4))
        plt.plot(user_shoulder_torsion, label='User Shoulder Torsion')
        plt.plot(pro_shoulder_torsion, label='Pro Shoulder Torsion', alpha=0.7)
        plt.title("Shoulder Torsion Angle Over Time")
        plt.ylabel("Angle (degrees)")
        plt.xlabel("Frame")
        plt.legend()
        plt.grid(True)
        plt.savefig("shoulder_torsion_angle.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Overlay plot user hip and shoulder torsion
        plt.figure(figsize=(10, 4))
        plt.plot(user_hip_torsion, label='User Hip Torsion')
        plt.plot(user_shoulder_torsion, label='User Shoulder Torsion', alpha=0.7)
        plt.title("User Hip and Shoulder Torsion Angles Over Time")
        plt.ylabel("Angle (degrees)")
        plt.xlabel("Frame")
        plt.legend()
        plt.grid(True)
        plt.savefig("user_hip_shoulder_torsion_overlay.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Throwing arm angular velocity plots (user vs pro)
        if throwing_arm_user_mean_angle is not None:
            plt.figure(figsize=(10, 4))
            plt.plot(throwing_arm_user_mean_angle, label='User Mean Throwing Arm Rotational Speed')
            plt.plot(throwing_arm_pro_mean_angle, label='Pro Mean Throwing Arm Rotational Speed', alpha=0.7)
            plt.title("Throwing Arm (Right) Mean Rotational Speed Over Time")
            plt.ylabel("Rotational Speed (degrees/second)")
            plt.xlabel("Frame")
            plt.legend()
            plt.grid(True)
            plt.savefig("throwing_arm_angular_velocity.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Biomechanical features spider chart
        labels = list(biomech_scores.keys())
        values = [v if v is not None else 0 for v in biomech_scores.values()]
        values += values[:1]  # close loop

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.set_ylim(0, 1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title("Biomechanical Similarity Scores")
        ax.grid(True)
        plt.savefig("biomechanical_similarity_spider_chart.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Overall similarity score as percentage bar chart
        plt.figure(figsize=(4, 6))
        plt.bar(['Overall Similarity'], [overall_score * 100], color='green')
        plt.ylim(0, 100)
        plt.ylabel('Similarity (%)')
        plt.title('Overall Biomechanical Similarity Score')
        plt.grid(axis='y')
        plt.savefig("overall_similarity_score_bar_chart.png", dpi=300, bbox_inches='tight')
        plt.close()

    return {
        'joint_metrics': joint_metrics,
        'group_metrics': group_metrics,
        'biomech_maes': biomech_maes,
        'biomech_scores': biomech_scores,
        'overall_similarity_score': overall_score,
        'throwing_arm_user_mean_angle': throwing_arm_user_mean_angle,
        'throwing_arm_pro_mean_angle': throwing_arm_pro_mean_angle,
        'right_shoulder_abduction': (user_abduction, pro_abduction),
        'hip_torsion': (user_hip_torsion, pro_hip_torsion),
        'shoulder_torsion': (user_shoulder_torsion, pro_shoulder_torsion),
    }



def evaluation_as_text(results):
    output = []
    output.append("Joint Metrics (Mean Absolute Error):")
    for j, m in results['joint_metrics'].items():
        output.append(f"  {m['name']:15s} | Dist MAE: {fmt_val(m['dist_mae'])} meters, "
                      f"Angle MAE: {fmt_val(m['angle_mae'])}°, "
                      f"Rotational Speed MAE: {fmt_val(m['ang_vel_mae'])} °/s")

    output.append("\nGroup Metrics (Mean Absolute Error):")
    for group, m in results['group_metrics'].items():
        output.append(f"  {group:10s} | Dist MAE: {fmt_val(m['dist_mae'])} meters, "
                      f"Angle MAE: {fmt_val(m['angle_mae'])}°, "
                      f"Rotational Speed MAE: {fmt_val(m['ang_vel_mae'])} °/s")

    output.append("\nBiomechanical Features (MAE and similarity score):")
    for k in results['biomech_maes'].keys():
        output.append(f"  {k:30s} | MAE: {fmt_val(results['biomech_maes'][k])} | "
                      f"Similarity Score: {fmt_val(results['biomech_scores'][k])}")

    output.append(f"\nOverall Biomechanical Similarity Score: {fmt_val(results['overall_similarity_score'] * 100)}%")
    
    return "\n".join(output)



if __name__ == "__main__":
    pro_kpts = np.load("./pro_3D_keypoints.npy")
    user_kpts = np.load("./user_3D_keypoints.npy")
    results = extended_evaluation(user_kpts, pro_kpts, plot=True)
    text_output = evaluation_as_text(results)
    print(text_output)
    # print("Extended evaluation completed. Results:")
    # import pprint
    # pprint.pprint(results)