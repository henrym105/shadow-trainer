import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import time

CUR_DIR = os.path.dirname(os.path.dirname(__file__))

# Load keypoints from .npz file
@st.cache_data(show_spinner=False)
def load_keypoints(npz_path):
    data = np.load(npz_path)
    print(f"Loaded keypoints data with shape: {data.shape}")
    return data

KEYPOINTS_DIR = os.path.join(CUR_DIR, "api_backend", "tmp_api_output", "henry1_full_output", "raw_keypoints")
user_data = load_keypoints(os.path.join(KEYPOINTS_DIR, "user_3D_keypoints.npy"))
pro_data = load_keypoints(os.path.join(KEYPOINTS_DIR, "pro_3D_keypoints.npy"))

st.set_page_config(layout="wide")
st.title("3D Human Pose Visualization")

# Session state for frame control
if "frame" not in st.session_state:
    st.session_state['frame'] = 0

# Controls
col1, col2, col3 = st.columns([1, 2, 6])

with col1:
    if st.button("⏮️"):
        st.session_state.frame = max(0, st.session_state.frame - 1)
with col2:
    if st.button("⏭️"):
        st.session_state.frame = min(len(user_data) - 1, st.session_state.frame + 1)
with col3:
    frame = st.slider("Frame", 0, len(user_data) - 1, st.session_state.frame)
    st.session_state.frame = frame


# Each pair is a connection (line between keypoints)
skeleton = [
    (0, 1), (1, 2), (2, 3),          # right leg
    (0, 4), (4, 5), (5, 6),          # left leg
    (0, 7), (7, 8), (8, 9), (9, 10), # spine to head
    (8, 11), (11, 12), (12, 13),     # left arm
    (8, 14), (14, 15), (15, 16)      # right arm
]


def plot_frame(frame_idx):
    fig = go.Figure()
    # Collect all keypoints to determine global min/max for cube aspect
    all_points = []
    for data in [user_data, pro_data]:
        frame_data = data[frame_idx]
        all_points.append(frame_data)

    all_points = np.concatenate(all_points, axis=0)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    z_min, z_max = np.min(all_points[:, 2]), np.max(all_points[:, 2])

    # Find the center and max range for cube
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    half_range = max_range / 2

    x_range = [x_center - half_range, x_center + half_range]
    y_range = [y_center - half_range, y_center + half_range]
    z_range = [z_center - half_range, z_center + half_range]

    plot_3d_skeleton(fig, user_data, frame_idx, color='blue')
    plot_3d_skeleton(fig, pro_data, frame_idx, color='gray')

    fig.update_layout(
        scene_camera=dict(eye=dict(x=1.25, y=1.25, z=0.8)),
        scene_dragmode='turntable',
        scene=dict(
            xaxis=dict(title='X', range=x_range, backgroundcolor='white', showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title='Y', range=y_range, backgroundcolor='white', showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(title='Z', range=z_range, backgroundcolor='white', showgrid=False, zeroline=False, showticklabels=False),
            bgcolor='white',
            aspectmode='cube',
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    return fig



def plot_3d_skeleton(fig, data, frame_idx, color):
    frame_data = data[frame_idx]
    # ------- Skeleton Keypoints ------- 
    fig.add_trace(go.Scatter3d(
        x=frame_data[:, 0],
        y=frame_data[:, 1],
        z=frame_data[:, 2],
        mode='markers',
        marker=dict(size=5, color='dark'+color, opacity=0.5),
        name='Keypoints'
    ))
    # ------- Lines (Bones) between keypoints ------- 
    for i, j in skeleton:
        x = [frame_data[i, 0], frame_data[j, 0], None]
        y = [frame_data[i, 1], frame_data[j, 1], None]
        z = [frame_data[i, 2], frame_data[j, 2], None]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=color, width=6),
            showlegend=False
        ))


# Display plot
plot = plot_frame(st.session_state.frame)
st.plotly_chart(plot, use_container_width=True)
