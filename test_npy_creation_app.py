import streamlit as st
import os
import numpy as np
import tempfile
import subprocess
import sys
import shutil
from save_user_keypoints import save_and_upload_user_keypoints
from pose_viz import create_pose_video

st.set_page_config(page_title="Test user_keypoints.npy Creation", layout="centered")
st.title("Test user_keypoints.npy Creation and Location")

st.markdown("""
This app will:
- Process a video (local file only)
- Run the pose pipeline (simulate or call your inference)
- Save user_keypoints.npy
- Upload to S3 (if desired)
- Show the local and S3 locations of the .npy file
""")

# --- Video upload ---
video_file = st.file_uploader("Upload Your Video (.mp4 or .mov)", type=["mp4", "mov"])

# --- Model size and S3 options ---
model_size = st.selectbox("Model Size", ["xs", "s", "b", "l"], index=0)
upload_to_s3 = st.checkbox("Upload user_keypoints.npy to S3 after creation", value=False)
s3_output_video_url = st.text_input("S3 Output Video URL (for S3 upload, e.g. s3://bucket/path/video.mp4)")

if st.button("Process and Test .npy Creation"):
    if video_file is None:
        st.error("Please upload a video file.")
        st.stop()
    with st.spinner("Processing video and creating user_keypoints.npy..."):
        # Save video to temp dir
        tmp_dir = tempfile.mkdtemp()
        video_path = os.path.join(tmp_dir, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        # Simulate pipeline: create dummy user_keypoints.npy
        user_kpts = np.random.rand(100, 17, 3)  # Replace with real pipeline if available
        user_kpts_path = os.path.join(tmp_dir, "user_keypoints.npy")
        np.save(user_kpts_path, user_kpts)
        st.success(f"user_keypoints.npy created at: {user_kpts_path}")
        st.write(f"user_keypoints.npy shape: {user_kpts.shape}")
        # Optionally upload to S3
        s3_url = None
        if upload_to_s3 and s3_output_video_url:
            s3_url = save_and_upload_user_keypoints(user_kpts, s3_output_video_url)
            if s3_url:
                st.success(f"user_keypoints.npy uploaded to: {s3_url}")
            else:
                st.error("Failed to upload user_keypoints.npy to S3.")
        # Optionally show a pose video
        if st.checkbox("Create and show pose video from .npy?"):
            pose_video_path = os.path.join(tmp_dir, "pose_video.mp4")
            create_pose_video(user_kpts, output_video_path=pose_video_path, frame_folder=os.path.join(tmp_dir, "pose_frames"), fps=20)
            with open(pose_video_path, "rb") as f:
                st.video(f.read())
        # Cleanup
        if st.checkbox("Delete temp files after test?"):
            shutil.rmtree(tmp_dir)
            st.info("Temporary files deleted.")
        else:
            st.info(f"Temporary files remain at: {tmp_dir}")
