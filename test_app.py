import streamlit as st
import requests
import os
import subprocess
import time
import threading

# --------------------------------------------------------------------------
# CONSTANTS
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# --------------------------------------------------------------------------

st.set_page_config(page_title="Shadow Trainer Video Processor (Test)", layout="centered")

# Logo + Title
st.markdown("<h1 style='text-align: center; font-size: 3.5em;'>Shadow Trainer (Test)</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([3, 3, 3])
with col2:
    st.image("Shadow Trainer Logo.png")

st.markdown("---")
st.markdown("""
Upload a video file or provide an S3 path to process it using the Shadow Trainer backend API. The processed video will be displayed below.
""")


# API setup (use original /process_video/ endpoint and /)
API_URL = os.environ.get("SHADOW_TRAINER_API_URL", "http://localhost:8000/process_video/")
API_HEALTH_URL = API_URL.replace("/process_video/", "/")

api_log_lines = []
api_process = None

def stream_api_logs():
    global api_process
    if api_process is None:
        return
    for line in iter(api_process.stdout.readline, b""):
        decoded = line.decode(errors="replace").rstrip()
        print(f"[API] {decoded}")
        api_log_lines.append(decoded)

def ensure_api_running():
    global api_process
    try:
        resp = requests.get(API_HEALTH_URL, timeout=2)
        if resp.status_code == 200:
            return True
    except Exception:
        pass
    st.info("Starting backend API server...")
    api_process = subprocess.Popen([
        "python", os.path.join("api_inference", "run_api.py")
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    threading.Thread(target=stream_api_logs, daemon=True).start()
    for _ in range(20):
        try:
            resp = requests.get(API_HEALTH_URL, timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            time.sleep(1)
    st.error("Failed to start backend API server. Please check logs.")
    return False

if not ensure_api_running():
    st.stop()

# Input options
input_mode = st.radio("Input Type", ["Local File", "S3 Path"], horizontal=True)
model_size = st.selectbox("Model Size", ["xs", "s", "b", "l"], index=0)

# Handedness
handedness = st.radio("What is your dominant hand?", ["Right-handed", "Left-handed"])

# Pitch type
pitch_type = st.multiselect(
    "Select your preferred pitch type(s):",
    options=["FF", "SI", "SL", "CH"]
)




# Only test on Shane Bieber video for pitcher, user uploads their own video
# (UI message removed as pitcher comparison is now always fixed in backend)
pitcher_s3_path = "s3://shadow-trainer-dev/cropped_videos/BieberShaneR/FF/cropped_03587bf2-48bb-45f5-84b5-38eec993695e_0.mp4"

video_file = None

video_file = st.file_uploader("Upload Your Video (.mp4 or .mov)", type=["mp4", "mov"])
videos_dir = os.path.join(CUR_DIR, "api_inference", "videos")
if os.path.exists(videos_dir):
    video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith((".mp4", ".mov"))]
    if video_files:
        st.markdown("**Sample Videos:**")
        cols = st.columns(len(video_files))
        for idx, vid in enumerate(video_files):
            vid_path = os.path.join(videos_dir, vid)
            with cols[idx]:
                st.video(vid_path, format="video/mp4", width="stretch")
                st.caption(vid)



if st.button("Process Video"):
    with st.spinner("Processing video, please wait..."):
        response = None
        temp_video_path = None
        if video_file is not None:
            TMP_DIR = os.path.join(CUR_DIR, "api_inference", "tmp_upload")
            os.makedirs(TMP_DIR, exist_ok=True)
            temp_video_path = os.path.join(TMP_DIR, video_file.name)
            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())
            params = {
                "file": temp_video_path,
                "model_size": model_size,
                "handedness": handedness,
                "pitch_type": ",".join(pitch_type),
                "compare_pitcher": "Shane Bieber",
                "compare_pitcher_video": pitcher_s3_path,
                "compare_force_reprocess": False
            }
            try:
                response = requests.post(API_URL, params=params)
            finally:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        else:
            st.error("Please upload your video.")
            st.stop()



# --- Automatic 3D keypoint overlay generation if .npy paths are returned by backend ---

import numpy as np
import requests as req
from io import BytesIO
from pose_viz import create_pose_overlay_video, create_pose_video

if response is not None and response.status_code == 200:
    result = response.json()
    video_url = result.get("output_video_local_path") or result.get("output_video_s3_url")
    user_kpts_path = result.get("user_keypoints_npy")
    pitcher_kpts_path = result.get("pitcher_keypoints_npy")
    st.success("Video processed successfully!")
    if video_url:
        if video_url.startswith("s3://"):
            st.write(f"Output Video S3 URL: {video_url}")
        else:
            if os.path.exists(video_url):
                with open(video_url, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                st.write(f"Output Video Path: {video_url}")
                st.download_button(
                    label="Download Output Video",
                    data=video_bytes,
                    file_name=os.path.basename(video_url),
                    mime="video/mp4"
                )
            else:
                st.write(f"Output Video Path: {video_url}")
    else:
        st.error("No output video path returned by API.")

    # --- Automatic overlay if both .npy files exist ---
    st.markdown("---")
    st.markdown("### 3D Keypoint Overlay (Auto)")
    if user_kpts_path and pitcher_kpts_path:
        # Load user keypoints (local file)
        if os.path.exists(user_kpts_path):
            user_kpts = np.load(user_kpts_path)
        else:
            st.error(f"User keypoints file not found: {user_kpts_path}")
            user_kpts = None

        # Load pitcher keypoints (local or S3/HTTP)
        if pitcher_kpts_path.startswith('http'):
            try:
                resp = req.get(pitcher_kpts_path)
                resp.raise_for_status()
                pitcher_kpts = np.load(BytesIO(resp.content))
            except Exception as e:
                st.error(f"Failed to download pitcher keypoints from URL: {pitcher_kpts_path}\n{e}")
                pitcher_kpts = None
        elif os.path.exists(pitcher_kpts_path):
            pitcher_kpts = np.load(pitcher_kpts_path)
        else:
            st.error(f"Pitcher keypoints file not found: {pitcher_kpts_path}")
            pitcher_kpts = None

        if user_kpts is not None and pitcher_kpts is not None:
            overlay_path = os.path.join(CUR_DIR, "overlay_pose.mp4")
            create_pose_overlay_video(user_kpts, pitcher_kpts, output_video_path=overlay_path, frame_folder="overlay_frames", fps=20)
            st.markdown("**Overlay: Your 3D Keypoints vs. Pitcher**")
            st.video(overlay_path)
        else:
            st.info("Could not load both user and pitcher keypoints for overlay.")
    else:
        st.info("3D keypoints not found in API response. Please check backend or upload .npy files manually.")
else:
    try:
        error = response.json().get("error", "Unknown error.")
    except Exception:
        error = response.text
    st.error(f"API Error: {error}")
