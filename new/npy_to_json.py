import numpy as np
import json

def convert_keypoints_npy_to_json(npy_path: str, json_path: str = None):
    """
    Converts a (nframes, 17, 3) .npy file into a flattened JSON file.

    Args:
        npy_path (str): Path to the .npy file.
        json_path (str): Path where the .json will be saved.
    """
    json_path = json_path or npy_path.replace(".npy", ".json")
    keypoints = np.load(npy_path)

    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError(f"Expected shape (nframes, 17, 3), got {keypoints.shape}")

    # Flatten each frame to a 1D list of 51 floats
    flattened = keypoints.reshape((keypoints.shape[0], -1)).tolist()

    with open(json_path, "w") as f:
        json.dump(flattened, f)

    print(f"Saved {len(flattened)} frames to {json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert .npy keypoints to .json format.")
    parser.add_argument(
        "npy_path",
        type=str,
        help="Path to the input .npy file."
    )

    args = parser.parse_args()
    convert_keypoints_npy_to_json(args.npy_path)