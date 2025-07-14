import streamlit as st
import requests
import os
from os.path import join as pjoin

# --- Constants ---
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = pjoin(CUR_DIR, "assets")
API_URL = "http://localhost:8002/video/process"

# --- Page Config ---
st.set_page_config(
    page_title="Shadow Trainer | Create Your Shadow",
    page_icon=pjoin(ASSETS_DIR, "Shadow Trainer Logo.png"),
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Header ---
st.markdown("""
    <div style='text-align: center; margin-top: 2em;'>
        <img src='app/assets/Shadow Trainer Logo.png' width='180' style='margin-bottom: 1em;' />
        <h1 style='font-size: 3em; font-weight: 700; margin-bottom: 0.2em;'>Shadow Trainer</h1>
        <h3 style='font-weight: 400; color: #555;'>AI-Powered Motion Analysis</h3>
    </div>
    <hr style='margin-top: 2em; margin-bottom: 2em;'>
""", unsafe_allow_html=True)

# --- Main Section ---
def main():
    st.markdown("""
    <div style='text-align: center; font-size: 1.2em; margin-bottom: 2em;'>
        Enter your video S3 path and click <b>Create Your Shadow</b> to process your motion video with our AI backend.
    </div>
    """, unsafe_allow_html=True)

    default_s3 = "s3://shadow-trainer-prod/sample_input/henry-mini.mov"
    video_path = st.text_input("Video S3 Path", default_s3, help="Paste your S3 video path here.")
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        run_clicked = st.button("Create Your Shadow", use_container_width=True)

    if run_clicked:
        if not video_path:
            st.error("Please enter a video S3 path.")
        else:
            with st.spinner("Processing your video. This may take a minute..."):
                try:
                    payload = {"file": video_path, "model_size": "xs"}
                    response = requests.post(
                        API_URL,
                        headers={"accept": "application/json", "Content-Type": "application/json"},
                        json=payload,
                        timeout=1000
                    )
                    if response.status_code == 200:
                        st.success("Your shadow has been created!")
                        resp_json = response.json()
                        with st.expander("Show API Response JSON"):
                            st.json(resp_json)
                        output_url = resp_json.get("output_video_s3_url")
                        output_local_path = resp_json.get("output_video_local_path")
                        if output_local_path and os.path.exists(output_local_path):
                            st.markdown("**Output Video Preview:**")
                            st.video(output_local_path)
                        elif output_url:
                            st.markdown(f"**Output Video S3 URL:** {output_url}")
                    else:
                        try:
                            st.error(f"API Error: {response.json().get('error', response.text)}")
                            with st.expander("Show API Error Response JSON"):
                                st.json(response.json())
                        except Exception:
                            st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    st.markdown("""
    <div style='margin-top: 3em; text-align: center; color: #888;'>
        &copy; 2025 Shadow Trainer. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
