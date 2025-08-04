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
Use the raw data and draw inference about how the user is deviating to the motion capture data of \
the pro. Be concise and blunt and professional, keep it to a few sentences. \

You may use baseball specific lingo when appropriate, you are speaking directly to the athlete. \
Use words Leading when a users motion is ahead of the pro, and "lagging" when the user is trailing behind the pro, or slower or later than the pro. \
Avoid saying generic things like "Work on your hip rotation mechanics to align more closely with the professional's" or "Overall, strive for better timing", \
instead, be specific about the differences and how the user can change to be more like the pro. 

Explain what the pro does, then explain what the user does and how that is different. 

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


def format_movement_analysis_for_llm(user_keypoints: np.ndarray, pro_keypoints: np.ndarray, is_lefty: bool = False) -> str:
    """
    Format body part direction deltas analysis into a single text string for LLM coaching.
    
    Args:
        user_keypoints: 3D keypoints for the user
        pro_keypoints: 3D keypoints for the professional
        is_lefty: If True, the left side is the throwing side
        
    Returns:
        Formatted text string containing all analysis data
    """
    analysis_parts = []
    
    # Add handedness context for the LLM
    handedness = "left-handed" if is_lefty else "right-handed"
    analysis_parts.append(f"ATHLETE HANDEDNESS: {handedness.upper()}")
    analysis_parts.append("")
    
    # Hip rotation analysis
    hip_deltas = body_part_direction_deltas(user_keypoints, "hips", pro_keypoints, "hips", is_lefty)
    analysis_parts.append("HIP ROTATION ANALYSIS:")
    analysis_parts.append("Hip Rotation, Cumulative (degrees):")
    analysis_parts.append("Note: These numbers represent how much the user hips have rotated relative to the pro hips at this point in time. Positive values indicate the user has rotated more than the pro at that point (ahead in rotation timing). Negative values indicate the user has rotated less than the pro (behind in rotation timing).")
    for i, delta in enumerate(hip_deltas):
        comp = "ahead of" if delta > 0 else "behind"
        analysis_parts.append(f"Frame {i}: {abs(int(delta))} degrees {comp} pro")
    analysis_parts.append("")
    
    # Shoulder rotation analysis
    shoulder_deltas = body_part_direction_deltas(user_keypoints, "shoulders", pro_keypoints, "shoulders", is_lefty)
    analysis_parts.append("SHOULDER ROTATION ANALYSIS:")
    analysis_parts.append("Shoulder Rotation, Cumulative (degrees):")
    analysis_parts.append("Note: Positive values indicate user has rotated shoulders more than the pro at that point (ahead in timing). Negative values indicate user has rotated shoulders less than the pro (behind in timing).")
    for i, delta in enumerate(shoulder_deltas):
        comp = "ahead of" if delta > 0 else "behind"
        analysis_parts.append(f"Frame {i}: {abs(int(delta))} degrees {comp} pro")
    analysis_parts.append("")
    
    # Hip vs shoulder separation (user and pro)
    user_separation = body_part_direction_deltas(user_keypoints, "shoulders", user_keypoints, "hips", is_lefty)
    pro_separation = body_part_direction_deltas(pro_keypoints, "shoulders", pro_keypoints, "hips", is_lefty)
    analysis_parts.append("HIP-SHOULDER SEPARATION COMPARISON:")
    analysis_parts.append("Torso Twist (degrees):")
    analysis_parts.append("Note: Positive values indicate hips have rotated more than shoulders (good separation - hips leading). Negative values indicate shoulders have rotated more than hips (poor separation - shoulders leading).")
    for i, (user_delta, pro_delta) in enumerate(zip(user_separation, pro_separation)):
        comp = "better than" if user_delta > pro_delta else "worse than"
        analysis_parts.append(f"Frame {i}: user={int(user_delta)}, pro={int(pro_delta)} --> User separation is {abs(int(user_delta - pro_delta))} degrees {comp} pro")
    analysis_parts.append("")

    return "\n".join(analysis_parts)


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
    
    return plot_paths




if __name__ == "__main__":
    user_npy = np.load("/home/ec2-user/shadow-trainer/api_backend/sample_videos/sample_output/raw_keypoints/user_3D_keypoints.npy")
    pro_npy = np.load("/home/ec2-user/shadow-trainer/api_backend/sample_videos/sample_output/raw_keypoints/pro_3D_keypoints.npy")

    # Generate formatted analysis text
    analysis_text = format_movement_analysis_for_llm(user_npy, pro_npy, is_lefty=False)
    
    print("FORMATTED ANALYSIS TEXT:")
    print("=" * 80)
    print(analysis_text)
    print("=" * 80)
    print()
    
    # Get LLM coaching feedback
    print("LLM COACHING FEEDBACK:")
    print("-" * 40)
    coaching_feedback = get_llm_coaching(analysis_text)
    print(coaching_feedback)
    print("-" * 40)


    plot_paths = generate_movement_analysis_plots(user_npy, pro_npy, "/home/ec2-user/shadow-trainer/api_backend/sample_videos/sample_output/plots", is_lefty=False)
    print("Generated plots:")
    for plot_type, path in plot_paths.items():
        print(f"{plot_type}: {path}")

