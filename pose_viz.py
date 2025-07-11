import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# --- 3D Pose Plotting and Video Functions ---
def show3Dpose(vals, ax, color='R'):
    ax.view_init(elev=15., azim=70)
    lcolor = (0, 0, 1)  # Blue
    rcolor = (1, 0, 0)  # Red
    if color == 'L':
        lcolor = (1, 1, 0)
        rcolor = (0, 1, 0)
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
    for i in range(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)
    RADIUS = 0.72
    RADIUS_Z = 0.7
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)

def create_pose_video(data, output_video_path="pose_video.mp4", frame_folder="pose_frames", fps=20):
    os.makedirs(frame_folder, exist_ok=True)
    for i, frame in enumerate(data):
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0], projection='3d')
        show3Dpose(frame, ax)
        frame_path = os.path.join(frame_folder, f"{i:04d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    img = cv2.imread(os.path.join(frame_folder, "0000.png"))
    height, width, _ = img.shape
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for i in range(len(data)):
        img = cv2.imread(os.path.join(frame_folder, f"{i:04d}.png"))
        video.write(img)
    video.release()
    print(f"Video saved to {output_video_path}")

def create_pose_overlay_video(data1, data2, output_video_path="pose_overlay.mp4", frame_folder="pose_overlay_frames", fps=20):
    os.makedirs(frame_folder, exist_ok=True)
    if len(data1) > len(data2):
        data1 = data1[:len(data2)]
    elif len(data2) > len(data1):
        data2 = data2[:len(data1)]
    for i, (frame1, frame2) in enumerate(zip(data1, data2)):
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0], projection='3d')
        show3Dpose(frame1, ax)
        show3Dpose(frame2, ax, 'L')
        frame_path = os.path.join(frame_folder, f"{i:04d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    img = cv2.imread(os.path.join(frame_folder, "0000.png"))
    height, width, _ = img.shape
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for i in range(len(data1)):
        img = cv2.imread(os.path.join(frame_folder, f"{i:04d}.png"))
        video.write(img)
    video.release()
    print(f"Overlay video saved to {output_video_path}")

# --- Utility Functions for Keypoints Manipulation ---
def resample_pose_sequence(pose_seq: np.ndarray, target_len: int) -> np.ndarray:
    n_frames, n_joints, n_dims = pose_seq.shape
    if n_frames == target_len:
        return pose_seq.copy()
    original_indices = np.linspace(0, n_frames - 1, num=n_frames)
    target_indices = np.linspace(0, n_frames - 1, num=target_len)
    resampled = np.zeros((target_len, n_joints, n_dims), dtype=np.float32)
    for j in range(n_joints):
        for d in range(n_dims):
            resampled[:, j, d] = np.interp(target_indices, original_indices, pose_seq[:, j, d])
    return resampled

def rotate_along_z(kpts: np.ndarray, degrees: float) -> np.ndarray:
    assert kpts.shape == (17, 3), "Input must be of shape (17, 3)"
    radians = np.deg2rad(degrees)
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)
    Rz = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,               0,    1]
    ])
    rotated_kpts = kpts @ Rz.T
    return rotated_kpts

def recenter_on_left_ankle(pose_array, target_xyz=(0.0, 0.0, 0.0)):
    left_ankle = pose_array[:, 6:7, :]
    target = np.array(target_xyz).reshape(1, 1, 3)
    return (pose_array - left_ankle + target)

def recenter_on_right_ankle(pose_array):
    right_ankle = pose_array[:, 3:4, :]
    return pose_array - right_ankle
