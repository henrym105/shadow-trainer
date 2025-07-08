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

st.set_page_config(page_title="Shadow Trainer Video Processor", layout="centered")

# Logo + Title
st.markdown("<h1 style='text-align: center; font-size: 3.5em;'>Shadow Trainer</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([3, 3, 3])
with col2:
    st.image(os.path.join(CUR_DIR, "Shadow Trainer Logo.png"))

st.markdown("---")
st.markdown("""
Upload a video file or provide an S3 path to process it using the Shadow Trainer backend API. The processed video will be displayed below.
""")

# API setup
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

# New: user handedness & pitch type
handedness = st.radio("What is your dominant hand?", ["Right-handed", "Left-handed"])
pitch_type = st.multiselect(
    "Select your preferred pitch type(s):",
    options=["FF", "SI", "SL", "CH"]
)

video_file = None
s3_path = None


if input_mode == "Local File":
    video_file = st.file_uploader("Upload Video (.mp4 or .mov)", type=["mp4", "mov"])

    # Show previews of videos in the videos/ directory as a grid (1-4 per row)
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
else:
    s3_path = st.text_input("Enter S3 Path (e.g., s3://bucket/key/video.mp4)")


if st.button("Process Video"):
    with st.spinner("Processing video, please wait..."):
        response = None
        temp_video_path = None
        if input_mode == "Local File" and video_file is not None:
            TMP_DIR = os.path.join(CUR_DIR, "api_inference", "tmp_upload")
            os.makedirs(TMP_DIR, exist_ok=True)
            temp_video_path = os.path.join(TMP_DIR, video_file.name)

            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())
            params = {
                "file": temp_video_path,
                "model_size": model_size,
                "handedness": handedness,
                "pitch_type": ",".join(pitch_type)
            }
            try:
                response = requests.post(API_URL, params=params)
            finally:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        elif input_mode == "S3 Path" and s3_path:
            params = {
                "file": s3_path,
                "model_size": model_size,
                "handedness": handedness,
                "pitch_type": ",".join(pitch_type)
            }
            response = requests.post(API_URL, params=params)
        else:
            st.error("Please provide a valid input.")
            st.stop()

        if response is not None and response.status_code == 200:
            result = response.json()
            video_url = result.get("output_video_local_path") or result.get("output_video_s3_url")
            if video_url:
                st.success("Video processed successfully!")
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
        else:
            try:
                error = response.json().get("error", "Unknown error.")
            except Exception:
                error = response.text
            st.error(f"API Error: {error}")

