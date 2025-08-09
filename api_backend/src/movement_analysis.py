"""
movement_analysis.py

NOTES about motion 
- pro and user keypoints are aligned at the start of the motion so that hips are facing the camera. 
- these keypoints are representations of a human body in 3d space throwing a baseball
- Use phrases like "throwing side" and "non-throwing side" to refer to the appropriate sides of the body based on the 
    value of is_lefty: bool. throwing side is left if is_lefty is True, otherwise it is right (right by default).
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import openai
import matplotlib.pyplot as plt

API_ROOT_DIR = Path(__file__).parent.parent

JOINT_NAMES = {
    'Pelvis': 0, 
    'Left Hip': 1, 
    'Left Knee': 2, 
    'Left Ankle': 3, 
    'Right Hip': 4, 
    'Right Knee': 5,
    'Right Ankle': 6, 
    'Spine': 7, 
    'Thorax': 8, 
    'Neck': 9,
    'Head': 10, 
    'Left Shoulder': 11, 
    'Left Elbow': 12,
    'Left Wrist': 13, 
    'Right Shoulder': 14, 
    'Right Elbow': 15, 
    'Right Wrist': 16
}

BASEBALL_LLM_PROMPT = """
You are a sports analyst specializing in baseball pitching. You have the following data comparing an \
athlete to a professional, and it is your job to give them advice on how to be more like the pro. \
Use the raw data and draw inference about how the user is deviating from the motion of \
the pro. Be concise and blunt and professional, keep it to a few sentences. \
You may use baseball specific lingo when appropriate, you are speaking directly to the athlete. \

Avoid saying generic things like "Work on your hip rotation mechanics to align more closely with the professional's" or "Overall, strive for better timing", \
instead, be specific about the differences and how the user can change to be more like the pro. 

Explain what the pro does, then explain what the user does and how that is different. 

Be very diligent and attentive to which numbers correspond to the user and which correspond to the pro. These results are worthless if you get those confused.
Watch your hyperbole when describing how much the user is ahead of or behind the pro in each part. Be specific and concise.

The data is as follows:
{raw_data}
"""

def body_part_direction_deltas(user_keypoints: np.ndarray, user_body_part: str, reference_keypoints: np.ndarray, reference_body_part: str = "hips", is_lefty: bool = False) -> list:
    """Calculate the difference in direction between user and professional keypoints for each frame of the video.
    
    Args:
        user_keypoints (np.ndarray): 3D keypoints for the user
        user_body_part (str): Body part to use for user's angles, e.g., "hips", "shoulders"
        reference_keypoints (np.ndarray): 3D keypoints for the reference body. This is the "ground truth" 
            pose against which the user is compared. Typically the professional's keypoints.
        reference_body_part (str): Body part to use for reference body's angles, default is "hips"
        is_lefty (bool): If True, expects counterclockwise rotation; if False, expects clockwise rotation

    Returns:
        list: Difference in cumulative rotation angles for each frame
    """
    assert user_keypoints.shape == reference_keypoints.shape, "User and professional keypoints must have the same shape"
    
    # Get cumulative angles for both user and pro
    user_angles = get_body_part_cum_rotation(user_keypoints, user_body_part, is_lefty)
    pro_angles = get_body_part_cum_rotation(reference_keypoints, reference_body_part, is_lefty)

    # Calculate differences
    angle_deltas = []
    for user_angle, pro_angle in zip(user_angles, pro_angles):
        diff = user_angle - pro_angle
        
        # Reverse sign for lefties since they rotate counterclockwise
        if is_lefty:
            diff = -diff
            
        angle_deltas.append(diff)

    return angle_deltas


def get_body_part_rotation_speed(keypoints: np.ndarray, body_part: str = "hips", is_lefty: bool = False, fps: float = 30.0) -> list:
    """Calculate frame-by-frame rotation speeds for a body part.
    
    Args:
        keypoints: 3D keypoints for a single person
        body_part: The body part to use for speed calculation, default is "hips"
        is_lefty: If True, expects counterclockwise rotation; if False, expects clockwise rotation
        fps: Frames per second to convert degrees/frame to degrees/second (default: 30.0)
        
    Returns:
        list: Rotation speeds in degrees/second for each frame
    """
    # Get cumulative rotation angles
    cum_angles = get_body_part_cum_rotation(keypoints, body_part, is_lefty)
    
    if len(cum_angles) <= 1:
        return [0.0] * len(cum_angles)
    
    speeds = [0.0]  # First frame has no speed (no previous frame to compare)
    
    # Calculate frame-by-frame speed (difference between consecutive cumulative angles)
    for i in range(1, len(cum_angles)):
        degrees_per_frame = cum_angles[i] - cum_angles[i-1]
        degrees_per_second = degrees_per_frame * fps
        speeds.append(degrees_per_second)
    
    return speeds


def get_body_part_cum_rotation(keypoints: np.ndarray, body_part: str = "hips", is_lefty: bool = False) -> list:
    """Calculate cumulative rotation angles for a single person's keypoints.
    
    Args:
        keypoints: 3D keypoints for a single person
        body_part: The body part to use for angle calculation, default is "hips"
        is_lefty: If True, expects counterclockwise rotation; if False, expects clockwise rotation
        
    Returns:
        list: Cumulative rotation angles for each frame
    """
    assert keypoints.ndim == 3 and keypoints.shape[2] == 3, "Keypoints should be in shape (num_frames, num_joints, 3)"
    
    num_frames = keypoints.shape[0]
    cumulative_angles = []
    
    # Get starting angle to calculate cumulative rotation
    start_angle = get_stance_angle(keypoints[0], use_body_part=body_part, is_lefty=is_lefty)
    
    cumulative = 0
    prev_angle = start_angle
    
    for i in range(num_frames):
        angle = get_stance_angle(keypoints[i], use_body_part=body_part, is_lefty=is_lefty)
        
        if i > 0:
            # Calculate frame-to-frame rotation using trigonometry to handle wraparound
            frame_diff = np.degrees(np.arctan2(np.sin(np.radians(angle - prev_angle)), 
                                              np.cos(np.radians(angle - prev_angle))))
            
            # For lefties, clockwise rotation is positive, so reverse the sign
            if is_lefty:
                frame_diff = -frame_diff
                
            cumulative += frame_diff
        
        cumulative_angles.append(cumulative)
        prev_angle = angle

    return cumulative_angles



def get_stance_angle(data: np.ndarray, use_body_part: str = "feet", is_lefty: bool = False) -> float:
    """Get the angle between the left and right ankles, hips, or shoulders from the 3D pose data.

    Args:
        data (np.ndarray): 3D pose data for a single frame, expected shape (17, 3), where the 2nd dimension contains x, y, z coordinates.
        use_body_part (str): Which body part to use for angle calculation: "feet", "hips", or "shoulders".
        is_lefty (bool): If True, the left side is considered the throwing side; otherwise, the right side is.

    Returns:
        float: Angle in degrees between the specified keypoints.
    """
    assert data.shape == (17, 3), f"Expected data shape (17, 3), got {data.shape}"

    if use_body_part == "hips":
        left = data[JOINT_NAMES["Left Hip"]][:2]
        right = data[JOINT_NAMES["Right Hip"]][:2]
    elif use_body_part == "shoulders":
        left = data[JOINT_NAMES["Left Shoulder"]][:2]
        right = data[JOINT_NAMES["Right Shoulder"]][:2]
    else:  # Default to "feet"
        left = data[JOINT_NAMES["Left Ankle"]][:2]
        right = data[JOINT_NAMES["Right Ankle"]][:2]

    assert len(left) == 2 and len(right) == 2, f"Expected left and right to have length 2, got {len(left)} and {len(right)}"

    return find_angle(right, left)


def find_angle(coord1: tuple, coord2: tuple) -> float:
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
        return 0  # Points are the same, angle is undefined

    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle


def get_summary_stats_text(user_data, pro_data, is_lefty: bool = False) -> str:
    """
    Generate summary statistics text comparing user and pro keypoints.
    
    Args:
        user_data: 3D keypoints for the user
        pro_data: 3D keypoints for the professional
        is_lefty: If True, the left side is the throwing side

    Returns:
        str: Formatted summary statistics text
    """
    # Calculate hip rotation deltas
    hip_deltas = body_part_direction_deltas(user_data, "hips", pro_data, "hips", is_lefty)
    max_hip = max(hip_deltas)
    min_hip = min(hip_deltas)
    max_hip_idx = hip_deltas.index(max_hip)
    min_hip_idx = hip_deltas.index(min_hip)
    percent_hip_ahead = 100 * sum(u > p for u, p in zip(get_body_part_cum_rotation(user_data, "hips", is_lefty), get_body_part_cum_rotation(pro_data, "hips", is_lefty))) / len(hip_deltas)

    # Calculate shoulder rotation deltas
    shoulder_deltas = body_part_direction_deltas(user_data, "shoulders", pro_data, "shoulders", is_lefty)
    max_shoulder = max(shoulder_deltas)
    min_shoulder = min(shoulder_deltas)
    max_shoulder_idx = shoulder_deltas.index(max_shoulder)
    min_shoulder_idx = shoulder_deltas.index(min_shoulder)
    percent_shoulder_ahead = 100 * sum(u > p for u, p in zip(get_body_part_cum_rotation(user_data, "shoulders", is_lefty), get_body_part_cum_rotation(pro_data, "shoulders", is_lefty))) / len(shoulder_deltas)

    # Format summary text
    summary = []
    summary.append(f"HIP ROTATION DELTAS: max={max_hip:.2f} (frame {max_hip_idx}), min={min_hip:.2f} (frame {min_hip_idx})")
    summary.append(f"User ahead of pro in hip rotation {percent_hip_ahead:.1f}% of frames.")
    summary.append(f"SHOULDER ROTATION DELTAS: max={max_shoulder:.2f} (frame {max_shoulder_idx}), min={min_shoulder:.2f} (frame {min_shoulder_idx})")
    summary.append(f"User ahead of pro in shoulder rotation {percent_shoulder_ahead:.1f}% of frames.")
    return "\n".join(summary)


def format_movement_analysis_for_llm(user_keypoints: np.ndarray, pro_keypoints: np.ndarray, is_lefty: bool = False) -> str:
    """
    Format body part direction deltas analysis into a JSON object text string for LLM coaching.
    
    Args:
        user_keypoints: 3D keypoints for the user
        pro_keypoints: 3D keypoints for the professional
        is_lefty: If True, the left side is the throwing side
        
    Returns:
        JSON formatted text string containing all analysis data
    """
    import json
    
    # Calculate all the analysis data
    hip_deltas = body_part_direction_deltas(user_keypoints, "hips", pro_keypoints, "hips", is_lefty)
    shoulder_deltas = body_part_direction_deltas(user_keypoints, "shoulders", pro_keypoints, "shoulders", is_lefty)
    user_hip_speeds = get_body_part_rotation_speed(user_keypoints, "hips", is_lefty)
    pro_hip_speeds = get_body_part_rotation_speed(pro_keypoints, "hips", is_lefty)
    user_shoulder_speeds = get_body_part_rotation_speed(user_keypoints, "shoulders", is_lefty)
    pro_shoulder_speeds = get_body_part_rotation_speed(pro_keypoints, "shoulders", is_lefty)
    user_separation = body_part_direction_deltas(user_keypoints, "shoulders", user_keypoints, "hips", is_lefty)
    pro_separation = body_part_direction_deltas(pro_keypoints, "shoulders", pro_keypoints, "hips", is_lefty)
    
    # Build the JSON structure
    data = {
        "athlete_handedness": "left" if is_lefty else "right",
        "hip_rotation": {
            "definition": "Comparison of how far the user has rotated their hips compared to the pro at that frame of the video, relative to their starting hip position. Positive = user's hips are currently rotated more than the pro, Negative = behind pro",
            "cumulative_degrees_rotated": [
                {"frame": i, "delta_deg": round(delta, 2)} for i, delta in enumerate(hip_deltas)
            ],
            "summary": {
                "max_delta_deg": round(max(hip_deltas), 2),
                "max_frame": hip_deltas.index(max(hip_deltas)),
                "min_delta_deg": round(min(hip_deltas), 2),
                "min_frame": hip_deltas.index(min(hip_deltas)),
                "percent_ahead": round(100 * sum(1 for delta in hip_deltas if delta > 0) / len(hip_deltas), 1)
            }
        },
        "shoulder_rotation": {
            "definition": "Comparison of how far the user has rotated their shoulders compared to the pro at that frame of the video, relative to their starting shoulder position. Positive = user's shoulders are currently rotated more than the pro, Negative = behind pro",
            "cumulative_degrees_rotated": [
                {"frame": i, "delta_deg": round(delta, 2)} for i, delta in enumerate(shoulder_deltas)
            ],
            "summary": {
                "max_delta_deg": round(max(shoulder_deltas), 2),
                "max_frame": shoulder_deltas.index(max(shoulder_deltas)),
                "min_delta_deg": round(min(shoulder_deltas), 2),
                "min_frame": shoulder_deltas.index(min(shoulder_deltas)),
                "percent_ahead": round(100 * sum(1 for delta in shoulder_deltas if delta > 0) / len(shoulder_deltas), 1)
            }
        },
        "hip_rotation_speed": {
            "definition": "Frame-by-frame rotation speeds for hip rotation in degrees per second",
            "user_speeds_deg_per_sec": [round(speed, 2) for speed in user_hip_speeds],
            "pro_speeds_deg_per_sec": [round(speed, 2) for speed in pro_hip_speeds],
            "speed_differences": [
                {"frame": i, "user_speed": round(user_speed, 2), "pro_speed": round(pro_speed, 2), "difference": round(user_speed - pro_speed, 2)}
                for i, (user_speed, pro_speed) in enumerate(zip(user_hip_speeds, pro_hip_speeds))
            ]
        },
        "shoulder_rotation_speed": {
            "definition": "Frame-by-frame rotation speeds for shoulder rotation in degrees per second",
            "user_speeds_deg_per_sec": [round(speed, 2) for speed in user_shoulder_speeds],
            "pro_speeds_deg_per_sec": [round(speed, 2) for speed in pro_shoulder_speeds],
            "speed_differences": [
                {"frame": i, "user_speed": round(user_speed, 2), "pro_speed": round(pro_speed, 2), "difference": round(user_speed - pro_speed, 2)}
                for i, (user_speed, pro_speed) in enumerate(zip(user_shoulder_speeds, pro_shoulder_speeds))
            ]
        },
        "hip_shoulder_separation": {
            "definition": "Hip-shoulder separation comparison. Positive values indicate hips have rotated more than shoulders (good separation - hips leading). Negative values indicate shoulders have rotated more than hips (poor separation - shoulders leading).",
            "user_separation_deg": [round(sep, 2) for sep in user_separation],
            "pro_separation_deg": [round(sep, 2) for sep in pro_separation],
            "separation_comparison": [
                {"frame": i, "user_separation": round(user_sep, 2), "pro_separation": round(pro_sep, 2), "difference": round(user_sep - pro_sep, 2)}
                for i, (user_sep, pro_sep) in enumerate(zip(user_separation, pro_separation))
            ]
        }
    }
    
    return json.dumps(data, indent=2)


def get_llm_coaching(raw_data_text: str) -> str:
    """
    Generate LLM-powered baseball throwing feedback from analysis data.
    
    Args:
        analysis_text: Structured analysis text from format_data_for_llm()
        
    Returns:
        LLM-generated coaching feedback
    """
    # Setup API key from environment
    dotenv_path = API_ROOT_DIR / '.env'
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        return "Error: OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."
    
    try:
        openai.api_key = api_key
        prompt = BASEBALL_LLM_PROMPT.format(raw_data=raw_data_text)
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert baseball pitching coach providing biomechanical analysis and coaching feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            # max_tokens=800,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating LLM feedback: {str(e)}"


def create_comparison_plot_and_save(user_values: list, pro_values: list, title: str, output_folder: str, filename: str = None, xlabel: str = "Frame", ylabel: str = "Degrees") -> str:
    """Create a matplotlib plot comparing user and pro values and save it as a PNG file.
    
    Args:
        user_values: List of values for the user
        pro_values: List of values for the pro
        title: Title for the plot
        output_folder: Directory where the PNG file will be saved
        filename: Optional filename (without extension). If None, uses sanitized title
        xlabel: Label for x-axis (default: "Frame")
        ylabel: Label for y-axis (default: "Degrees")
        
    Returns:
        Path to the saved PNG file
    """
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        # Sanitize title for filename
        filename = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = filename.replace(' ', '_').lower()
    
    # Create the plot
    plt.figure(figsize=(6, 4))
    
    frames = range(len(user_values))
    plt.plot(frames, user_values, marker='o', linewidth=2, markersize=4, label='User', color='red')
    plt.plot(frames, pro_values, marker='s', linewidth=2, markersize=4, label='Pro', color="black")

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # Save the plot
    png_path = output_path / f"{filename}.png"
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return str(png_path)



def create_human_body_joint_plot(user_keypoints: np.ndarray, pro_keypoints: np.ndarray, output_folder: str, is_lefty: bool = False) -> str:
    """
    Create a human body visualization showing joint similarity scores arranged anatomically.
    
    Args:
        user_keypoints: 3D keypoints for the user
        pro_keypoints: 3D keypoints for the professional
        output_folder: Directory where the PNG file will be saved
        is_lefty: If True, the left side is the throwing side
        
    Returns:
        Path to the saved visualization image
    """
    # Calculate joint similarity scores
    joint_scores = calculate_joint_similarity_scores(user_keypoints, pro_keypoints)
    
    # Create figure with appropriate size
    _, ax = plt.subplots(figsize=(5, 7))
    # ax.set_xlim(-3, 3)
    # ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Define anatomical positions for each joint (x, y coordinates)
    # Negative x = left side, positive x = right side from viewer's perspective
    joint_positions = {
        'Head': (0, 3.7),
        'Neck': (0, 3.0),
        'Thorax': (0, 2.0),
        'Spine': (0, 1.0),
        'Pelvis': (0, 0),
        'Left Shoulder': (-1.2, 2.0),
        'Right Shoulder': (1.2, 2.0),
        'Left Elbow': (-1.8, 1.0),
        'Right Elbow': (1.8, 1.0),
        'Left Wrist': (-2.2, 0.0),
        'Right Wrist': (2.2, 0.0),
        'Left Hip': (-0.6, 0),
        'Right Hip': (0.6, 0),
        'Left Knee': (-0.8, -1.5),
        'Right Knee': (0.8, -1.5),
        'Left Ankle': (-0.8, -3.0),
        'Right Ankle': (0.8, -3.0)
    }
    
    # Color mapping for scores (0-10 scale)
    def score_to_color(score):
        # Red (poor) to yellow (okay) to green (excellent)
        if score <= 5.0:
            # Red to yellow transition
            ratio = score / 5.0
            return (1.0, ratio, 0.0)
        else:
            # Yellow to green transition
            ratio = (score - 5.0) / 5.0
            return (1.0 - ratio, 1.0, 0.0)
    
    # Draw joints as circles with color-coded scores
    joint_names_ordered = [name for name, _ in sorted(JOINT_NAMES.items(), key=lambda x: x[1])]
    
    for i, joint_name in enumerate(joint_names_ordered):
        if joint_name in joint_positions:
            x, y = joint_positions[joint_name]
            score = joint_scores[i]
            color = score_to_color(score)
            
            # Draw joint circle
            circle = plt.Circle((x, y), 0.2, color=color, alpha=0.8, zorder=2)
            ax.add_patch(circle)
            
            # Add score text
            # ax.text(x, y, f'{score:.1f}', ha='center', va='center', 
            ax.text(x, y, f'{int(score)}', ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='black', zorder=3)
            
            # Add joint label
            label_offset = 0.35
            if 'Left' in joint_name or joint_name in ['Head', 'Neck', 'Thorax', 'Spine', 'Pelvis']:
                label_y = y - label_offset
            else:
                label_y = y - label_offset
                
            ax.text(x, label_y, joint_name.replace('Left ', 'L ').replace('Right ', 'R '), 
                   ha='center', va='top', fontsize=7, color='black', zorder=3)
    
    # Draw body outline/skeleton for reference
    # Spine line
    ax.plot([0, 0], [0, 3.5], 'k-', alpha=0.3, linewidth=2, zorder=1)
    
    # Shoulder line
    ax.plot([-1.2, 1.2], [2.0, 2.0], 'k-', alpha=0.3, linewidth=2, zorder=1)
    
    # Hip line
    ax.plot([-0.6, 0.6], [0, 0], 'k-', alpha=0.3, linewidth=2, zorder=1)
    
    # Arms
    ax.plot([-1.2, -1.8, -2.2], [2.0, 1.0, 0.0], 'k-', alpha=0.3, linewidth=2, zorder=1)  # Left arm
    ax.plot([1.2, 1.8, 2.2], [2.0, 1.0, 0.0], 'k-', alpha=0.3, linewidth=2, zorder=1)   # Right arm
    
    # Legs
    ax.plot([-0.6, -0.8, -0.8], [0, -1.5, -3.0], 'k-', alpha=0.3, linewidth=2, zorder=1)  # Left leg
    ax.plot([0.6, 0.8, 0.8], [0, -1.5, -3.0], 'k-', alpha=0.3, linewidth=2, zorder=1)    # Right leg
    
    # Add title and legend
    ax.text(0, 4.2, 'Joint Similarity Scores', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Save the plot
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_filename = "joint_distance_comparison_plot.png"
    plot_path = output_path / plot_filename
    
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(plot_path)



def generate_movement_analysis_plots(user_keypoints: np.ndarray, pro_keypoints: np.ndarray, output_folder: str, is_lefty: bool = False) -> dict:
    """
    Generate all movement analysis plots and save them to the output folder.
    
    Args:
        user_keypoints: 3D keypoints for the user
        pro_keypoints: 3D keypoints for the professional
        output_folder: Directory where the PNG files will be saved
        is_lefty: If True, the left side is the throwing side
        
    Returns:
        Dictionary with plot types as keys and file paths as values
    """
    plot_paths = {}
    
    # Generate cumulative rotation data for plotting
    user_hip_angles = get_body_part_cum_rotation(user_keypoints, "hips", is_lefty)
    pro_hip_angles = get_body_part_cum_rotation(pro_keypoints, "hips", is_lefty)
    
    user_shoulder_angles = get_body_part_cum_rotation(user_keypoints, "shoulders", is_lefty)
    pro_shoulder_angles = get_body_part_cum_rotation(pro_keypoints, "shoulders", is_lefty)
    
    # Hip rotation comparison plot
    hip_plot_path = create_comparison_plot_and_save(
        user_hip_angles, 
        pro_hip_angles,
        "Hip Rotation Comparison: User vs Pro",
        output_folder,
        "hip_rotation",
        xlabel="Frame",
        ylabel="Cumulative Rotation (degrees)"
    )
    plot_paths["hip_rotation"] = hip_plot_path
    
    # Shoulder rotation comparison plot
    shoulder_plot_path = create_comparison_plot_and_save(
        user_shoulder_angles,
        pro_shoulder_angles,
        "Shoulder Rotation Comparison: User vs Pro", 
        output_folder,
        "shoulder_rotation",
        xlabel="Frame",
        ylabel="Cumulative Rotation (degrees)"
    )
    plot_paths["shoulder_rotation"] = shoulder_plot_path
    
    # Hip-shoulder separation comparison
    user_separation = body_part_direction_deltas(user_keypoints, "shoulders", user_keypoints, "hips", is_lefty)
    pro_separation = body_part_direction_deltas(pro_keypoints, "shoulders", pro_keypoints, "hips", is_lefty)
    
    separation_plot_path = create_comparison_plot_and_save(
        user_separation,
        pro_separation,
        "Hip-Shoulder Separation: User vs Pro",
        output_folder,
        "hip_shoulder_separation",
        xlabel="Frame", 
        ylabel="Separation (degrees)"
    )
    plot_paths["hip_shoulder_separation"] = separation_plot_path
    
    # Hip rotation speed comparison plot
    user_hip_speeds = get_body_part_rotation_speed(user_keypoints, "hips", is_lefty)
    pro_hip_speeds = get_body_part_rotation_speed(pro_keypoints, "hips", is_lefty)
    
    hip_speed_plot_path = create_comparison_plot_and_save(
        user_hip_speeds,
        pro_hip_speeds,
        "Hip Rotation Speed Comparison: User vs Pro",
        output_folder,
        "hip_rotation_speed",
        xlabel="Frame",
        ylabel="Rotation Speed (degrees/second)"
    )
    plot_paths["hip_rotation_speed"] = hip_speed_plot_path
    
    # Shoulder rotation speed comparison plot
    user_shoulder_speeds = get_body_part_rotation_speed(user_keypoints, "shoulders", is_lefty)
    pro_shoulder_speeds = get_body_part_rotation_speed(pro_keypoints, "shoulders", is_lefty)
    
    shoulder_speed_plot_path = create_comparison_plot_and_save(
        user_shoulder_speeds,
        pro_shoulder_speeds,
        "Shoulder Rotation Speed Comparison: User vs Pro",
        output_folder,
        "shoulder_rotation_speed",
        xlabel="Frame",
        ylabel="Rotation Speed (degrees/second)"
    )
    plot_paths["shoulder_rotation_speed"] = shoulder_speed_plot_path
    
    joint_comparison_plot_filename = create_human_body_joint_plot(user_keypoints, pro_keypoints, output_folder, is_lefty)
    joint_comparison_plot_path = Path(output_folder) / joint_comparison_plot_filename
    plot_paths["joint_distance_comparison_plot"] = str(joint_comparison_plot_path)

    return plot_paths


def calculate_joint_similarity_scores(user_keypoints: np.ndarray, pro_keypoints: np.ndarray) -> list:
    """
    Calculate similarity scores for each joint between user and pro keypoints.
    Higher scores indicate more similarity between user and pro.
    
    Args:
        user_keypoints: 3D keypoints for the user
        pro_keypoints: 3D keypoints for the professional
        
    Returns:
        List of similarity scores (0-10 scale) for each joint
    """
    num_keypoints = user_keypoints.shape[1]
    assert num_keypoints == 17
    
    # Calculate the overall difference between user and pro keypoints for each joint
    joint_overall_difference = []
    for i in range(num_keypoints):
        # Calculate the direct 3D distance between user and pro keypoints for this joint across all frames
        joint_differences = np.linalg.norm(user_keypoints[:, i, :] - pro_keypoints[:, i, :], axis=1)
        # Use mean difference across all frames for this joint
        joint_overall_difference.append(np.mean(joint_differences))

    # Normalize the differences to create scores (lower difference = higher score)
    max_difference = np.max(joint_overall_difference) if joint_overall_difference else 1.0
    # Add a small epsilon to max_difference to avoid division by zero if all differences are zero
    max_difference += 1e-8
    # Scale scores to a 0-10 range (inverted so lower difference = higher score)
    joint_scores = [max(0, 1 - (diff / max_difference)) * 10 for diff in joint_overall_difference]
    
    return joint_scores


def create_spider_plot_all_joints(user_keypoints: np.ndarray, pro_keypoints: np.ndarray, output_folder: str, is_lefty: bool = False) -> str:
    """
    Create a spider plot comparing all joints between user and pro keypoints.
    
    Args:
        user_keypoints: 3D keypoints for the user
        pro_keypoints: 3D keypoints for the professional
        output_folder: Directory where the PNG file will be saved
        is_lefty: If True, the left side is the throwing side

    Returns:
        Path to the saved spider plot image
    """
    # Calculate joint similarity scores using the new function
    joint_scores = calculate_joint_similarity_scores(user_keypoints, pro_keypoints)

    # Get the joint names for the labels
    joint_labels = [name for name, _ in sorted(JOINT_NAMES.items(), key=lambda x: x[1])]

    # Prepare data for the spider chart
    scores = np.array(joint_scores)

    # Shorten labels for better display
    labels = [l.replace("Left", "L.") for l in joint_labels]
    labels = [l.replace("Right", "R.") for l in labels]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    # Create the spider chart
    _, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, alpha=0.25, color='DarkGreen')

    # Keep angle gridlines but remove degree labels
    ax.set_xticks(angles)                # keep the lines
    ax.set_xticklabels([])               # remove default degree labels

    # Custom joint labels at padded distance, aligned with circle spots
    label_padding = 6
    for angle, label in zip(angles, labels):
        # Convert angle to degrees and adjust for text alignment
        rotation_angle = np.degrees(angle)
        # Adjust rotation so text aligns with the radial line
        if angle > np.pi / 2 and angle < 3 * np.pi / 2:
            # For angles in the left half, rotate 180 degrees more to keep text readable
            rotation_angle += 180
            ha = 'center'
        else:
            ha = 'center'
        
        ax.text(angle, label_padding, label,
                ha=ha, va='center', fontsize=10, rotation=rotation_angle)

    # Set chart title and limits
    ax.set_title("Overall Similarity Scores by Joint", va='bottom', fontsize=12, y=1.3)
    ax.set_ylim(0, 10)
    ax.grid(True)

    # Save the plot
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_filename = "joint_distance_comparison_plot.png"
    plot_path = output_path / plot_filename
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')

    plt.close()

    print("Spider Plot Joint Labels and Scores:")
    for label, score in zip(labels, scores):
        print(f"{label}: {score:.2f}")

    return plot_filename


if __name__ == "__main__":
    user_npy = np.load("/home/ec2-user/shadow-trainer/api_backend/sample_videos/sample_output/raw_keypoints/user_3D_keypoints.npy")
    pro_npy = np.load("/home/ec2-user/shadow-trainer/api_backend/sample_videos/sample_output/raw_keypoints/pro_3D_keypoints.npy")

    # Generate formatted analysis text
    analysis_text = format_movement_analysis_for_llm(user_npy, pro_npy, is_lefty=True)
    
    # Save analysis text as info.json
    output_folder = "/home/ec2-user/shadow-trainer/api_backend/sample_videos/sample_output"
    info_json_path = Path(output_folder) / "info.json"
    with open(info_json_path, 'w') as f:
        f.write(analysis_text)
    print(f"Analysis saved to: {info_json_path}")


    # print("FORMATTED ANALYSIS TEXT:")
    # print("=" * 80)
    # print(analysis_text)
    # print("=" * 80)
    # print()
    
    # Get LLM coaching feedback
    # print("LLM COACHING FEEDBACK:")
    # print("-" * 40)
    # coaching_feedback = get_llm_coaching(analysis_text)
    # print(coaching_feedback)
    # print("-" * 40)

    output_folder = "/home/ec2-user/shadow-trainer/api_backend/sample_videos/sample_output/plots"

    plot_paths = generate_movement_analysis_plots(user_npy, pro_npy, output_folder, is_lefty=False)
    print("Generated plots:")
    for plot_type, path in plot_paths.items():
        print(f"{plot_type}: {path}")


    create_human_body_joint_plot(user_npy, pro_npy, output_folder, is_lefty=True)

