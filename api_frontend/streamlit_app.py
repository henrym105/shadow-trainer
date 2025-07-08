
import streamlit as st
import requests
import os
import subprocess
import time

st.set_page_config(page_title="Shadow Trainer Video Processor", layout="centered")
st.title("Shadow Trainer Video Processor")

st.markdown("""
Upload a video file or provide an S3 path to process it using the Shadow Trainer backend API. The processed video will be displayed below.
""")


# Ensure API is running before proceeding
API_URL = os.environ.get("SHADOW_TRAINER_API_URL", "http://localhost:8000/process_video/")
API_HEALTH_URL = API_URL.replace("/process_video/", "/")

import threading

api_log_lines = []

def stream_api_logs():
    """Continuously read logs from the API process and print to terminal."""
    global api_process
    if api_process is None:
        return
    for line in iter(api_process.stdout.readline, b""):
        decoded = line.decode(errors="replace").rstrip()
        print(f"[API] {decoded}")
        api_log_lines.append(decoded)

api_process = None

def ensure_api_running():
    global api_process
    try:
        resp = requests.get(API_HEALTH_URL, timeout=2)
        if resp.status_code == 200:
            return True
    except Exception:
        pass
    st.info("Starting backend API server...")
    # Start the API in a subprocess and capture stdout/stderr
    api_process = subprocess.Popen([
        "python", os.path.join("api_inference", "run_api.py")
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    # Start a thread to stream logs
    threading.Thread(target=stream_api_logs, daemon=True).start()
    # Wait for it to be up
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

input_mode = st.radio("Input Type", ["Local File", "S3 Path"], horizontal=True)
model_size = st.selectbox("Model Size", ["xs", "s", "b", "l"], index=0)
video_file = None
s3_path = None

if input_mode == "Local File":
    video_file = st.file_uploader("Upload Video (.mp4 or .mov)", type=["mp4", "mov"])
else:
    s3_path = st.text_input("Enter S3 Path (e.g., s3://bucket/key/video.mp4)")

if st.button("Process Video"):
    with st.spinner("Processing video, please wait..."):
        if input_mode == "Local File" and video_file is not None:
            # Save uploaded file to a temp location
            temp_video_path = os.path.join("tmp_upload", video_file.name)
            os.makedirs("tmp_upload", exist_ok=True)
            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())
            params = {"file": temp_video_path, "model_size": model_size}
            try:
                response = requests.post(API_URL, params=params)
            finally:
                # Clean up temp file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        elif input_mode == "S3 Path" and s3_path:
            params = {"file": s3_path, "model_size": model_size}
            response = requests.post(API_URL, params=params)
        else:
            st.error("Please provide a valid input.")
            st.stop()

        if response.status_code == 200:
            result = response.json()
            video_url = result.get("output_video_local_path") or result.get("output_video_s3_url")
            if video_url:
                st.success("Video processed successfully!")
                if video_url.startswith("s3://"):
                    st.write(f"Output Video S3 URL: {video_url}")
                else:
                    # Try to display the video if local and exists
                    if os.path.exists(video_url):
                        with open(video_url, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                        st.write(f"Output Video Path: {video_url}")
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
