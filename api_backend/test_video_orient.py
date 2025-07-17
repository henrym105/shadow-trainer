import logging

def test_rotate_video_until_upright():
    """Test the rotate_video_until_upright function"""
    import sys
    import os

    # Add the src directory to path if needed
    sys.path.append('/home/ec2-user/shadow-trainer/api_backend')

    from src.yolo2d import rotate_video_until_upright

    # video_path = "/home/ec2-user/shadow-trainer/api_backend/sample_videos/upside_down_test.mov"
    video_path = "/home/ec2-user/shadow-trainer/api_backend/sample_videos/Left_Hand_Friend_Side.MOV"

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    print(f"Testing video orientation detection on: {video_path}")

    try:
        # Call the function
        num_rotations = rotate_video_until_upright(video_path, debug=False)
        print(f"Function completed successfully, rotated {num_rotations * 90} degrees to make the video upright.")

    except Exception as e:
        print(f"Error occurred during testing: {str(e)}")
        import traceback
        traceback.print_exc()

# Run the test
if __name__ == "__main__":
    print(test_rotate_video_until_upright())
