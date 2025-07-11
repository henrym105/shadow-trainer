# MVP Streamlit UI for Shadow Trainer
import streamlit as st
import requests
import os
from os.path import join as pjoin

# --- Constants ---
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = pjoin(CUR_DIR, "assets")
API_URL = "http://localhost:8002/video/process"
# API_URL = "http://www.shadow-trainer.com/api/video/process"

# --- Header ---
st.markdown("<h1 style='text-align: center; font-size: 3.5em;'>Shadow Trainer</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([3, 3, 3])
with col2:
    st.image(pjoin(ASSETS_DIR, "Shadow Trainer Logo.png"))
st.markdown("---")

# --- Input Section ---
st.markdown("""
Enter the path to your video (local path or S3 URI) and click Run to process it using the backend API.
""")

video_path = st.text_input("Video Path", "s3://shadow-trainer-prod/sample_input/henry-mini.mov")
run_clicked = st.button("Run")

# --- API Call and Response Display ---
if run_clicked:
    if not video_path:
        st.error("Please enter a video path.")
    else:
        with st.spinner("Processing video, please wait..."):
            try:
                # Mimic the cURL: POST with JSON body {"file": ..., "model_size": "xs"}
                payload = {"file": video_path, "model_size": "xs"}
                response = requests.post(
                    API_URL,
                    headers={"accept": "application/json", "Content-Type": "application/json"},
                    json=payload,
                    timeout=1000
                )
                print(f"RESPONSE: {response}") # Debugging line, print to terminal
                if response.status_code == 200:
                    st.success("API call successful!")
                    # Show the full JSON payload in an expandable section for clarity
                    with st.expander("Show API Response JSON"):
                        st.json(response.json())
                    # --- Show output_video_s3_url if present ---
                    output_url = response.json().get("output_video_s3_url")
                    output_local_path = response.json().get("output_video_local_path")
                    # if output_local_path and os.path.exists(output_local_path):
                    #     st.markdown("**Output Video Preview:**")
                    #     st.markdown(f"**Output Video Local Path:**\n\n{output_local_path}")
                    #     st.video(output_local_path)
                else:
                    try:
                        st.error(f"API Error: {response.json().get('error', response.text)}")
                        with st.expander("Show API Error Response JSON"):
                            st.json(response.json())
                    except Exception:
                        st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")


# if output_local_path and os.path.exists(output_local_path):
#     st.markdown("**Output Video Preview:**")
#     st.markdown(f"**Output Video Local Path:**\n\n{output_local_path}")
#     st.video(output_local_path)
st.markdown("**Output Video Preview:**")
st.markdown(f"**Output Video Local Path:**\n\n/home/ec2-user/shadow-trainer/api_backend/tmp_api_output/henry-mini_output/henry-mini.mp4")
st.video("/home/ec2-user/shadow-trainer/api_backend/tmp_api_output/henry-mini_output/henry-mini.mp4")

