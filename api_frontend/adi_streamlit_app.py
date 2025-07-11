import streamlit as st
import requests
import os
import subprocess
import time
import threading

from os.path import join as pjoin

# --------------------------------------------------------------------------
# CONSTANTS
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = pjoin(CUR_DIR, "assets")
# --------------------------------------------------------------------------

# st.set_page_config(page_title="Shadow Trainer Video Processor", layout="centered")

# Logo + Title
st.markdown("<h1 style='text-align: center; font-size: 3.5em;'>Shadow Trainer</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([3, 3, 3])
with col2:
    st.image(pjoin(ASSETS_DIR, "Shadow Trainer Logo.png"))

st.markdown("---")
st.markdown("""
Upload a video file or provide an S3 path to process it using the Shadow Trainer backend API. The processed video will be displayed below.
""")

# API setup
API_URL = "http://localhost:8002/"
API_HEALTH_URL = API_URL + "health/"
API_VIDEO_ENDPOINT = API_URL + "video/process"

api_log_lines = []
api_process = None


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


if st.button("Process Video"):
    with st.spinner("Processing video, please wait..."):
        response = None
        temp_video_path = None
        if input_mode == "Local File" and video_file is not None:
            TMP_DIR = pjoin(CUR_DIR, "tmp_upload")
            os.makedirs(TMP_DIR, exist_ok=True)
            temp_video_path = pjoin(TMP_DIR, video_file.name)

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

