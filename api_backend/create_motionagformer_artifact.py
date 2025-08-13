import torch
from torch import nn
import numpy as np

from src.model.MotionAGFormer import MotionAGFormer
from src.utils import get_config, download_file_if_not_exists, normalize_screen_coordinates
# Run this file from folder `shadow-trainer/api_backend`


def create_jit_trace_pt_file(yaml_path: str, model_size: str = 'xs', checkpoint_dir: str = "checkpoint", device: str = 'cpu'):
    # Load config first
    args = get_config(yaml_path)
    args = {k: v for k, v in args.items() if k in [
        'n_layers', 'dim_in', 'dim_feat', 'dim_rep', 'dim_out',
        'mlp_ratio', 'act_layer',
        'attn_drop', 'drop', 'drop_path',
        'use_layer_scale', 'layer_scale_init_value', 'use_adaptive_fusion',
        'num_heads', 'qkv_bias', 'qkv_scale',
        'hierarchical',
        'use_temporal_similarity', 'neighbour_num', 'temporal_connection_len',
        'use_tcn', 'graph_only',
        'n_frames'
    ]}
    args['act_layer'] = torch.nn.GELU
    
    # Create model with DataParallel wrapper like in inference.py
    model = nn.DataParallel(MotionAGFormer(**args)).to(device)
    
    # Load pre-trained weights
    model_filename = f"motionagformer-{model_size}-h36m*.pth*"
    model_path = download_file_if_not_exists(model_filename, checkpoint_dir)
    pre_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(pre_dict['model'], strict=True)
    model.eval()

    # Load example input for tracing - following the same preprocessing as get_pose3D_no_vis
    sample_data = np.load('sample_videos/sample_output/2D_keypoints.npy')
    
    # Use only first n_frames, 27 for xs model
    n_frames = args['n_frames']
    window = sample_data[:, :n_frames, :, :]  # shape: (1, n_frames, 17, 3)

    # Normalize coordinates (assuming 1920x1080 video size - you may need to adjust)
    input_2D = normalize_screen_coordinates(window, w=1920, h=1080)
    example_input = torch.from_numpy(input_2D.astype('float32')).to(device)
    
    # Create JIT model using tracing
    with torch.no_grad():
        model_artifact = torch.jit.trace(model, example_input)

    # Save the JIT model
    output_path = f'{checkpoint_dir}/motionagformer-{model_size}-jit-trace.pt'
    # output_path = f'./motionagformer_{model_size}_jit_trace.pt'
    torch.jit.save(model_artifact, output_path)
    return output_path


def load_and_use_jit_model(model_path: str, device: str = 'cpu'):
    jit_model = torch.jit.load(model_path, map_location=device)
    jit_model.eval()
    
    # Example inference using the same sample data preprocessing
    with torch.no_grad():
        sample_data = np.load('sample_videos/sample_output/2D_keypoints.npy')
        window = sample_data[:, :27, :, :]  # Extract x,y coordinates
        input_2D = normalize_screen_coordinates(window, w=1920, h=1080)
        test_input = torch.from_numpy(input_2D.astype('float32')).to(device)
        predictions = jit_model(test_input)
        
    return predictions


# Example usage
if __name__ == "__main__":
    yaml_config_path = "src/configs/h36m/MotionAGFormer-xsmall.yaml"
    OUTPUT_PATH = f'checkpoint/motionagformer-xs-jit-trace.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create and save JIT model
    jit_model_path = create_jit_trace_pt_file(yaml_config_path, model_size='xs', device=device)
    print(f"Model saved to: {jit_model_path}")
    
    # Load and test the model
    predictions = load_and_use_jit_model(jit_model_path, device=device)
    print(f"Prediction shape: {predictions.shape}")
