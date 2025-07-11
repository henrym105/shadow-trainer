import numpy as np
import boto3
import os
from urllib.parse import urlparse
from botocore.exceptions import NoCredentialsError

def upload_to_s3(local_file, s3_bucket, s3_key, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    if aws_access_key_id and aws_secret_access_key:
        s3 = boto3.client('s3',
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key,
                          region_name=region_name)
    else:
        s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file, s3_bucket, s3_key)
        s3_url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
        print(f"Uploaded {local_file} to {s3_url}")
        return s3_url
    except NoCredentialsError:
        print("AWS credentials not available.")
        return None
    finally:
        if os.path.exists(local_file):
            os.remove(local_file)

def save_and_upload_user_keypoints(user_keypoints: np.ndarray, output_video_s3_url: str, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    """
    Saves user_keypoints as user_keypoints.npy and uploads to the same S3 folder as the output video.
    output_video_s3_url: e.g. 's3://my-bucket/outputs/user123/output_video.mp4'
    Returns the S3 URL of the uploaded .npy file.
    """
    parsed = urlparse(output_video_s3_url)
    if parsed.scheme != "s3":
        raise ValueError("output_video_s3_url must be an S3 URL (s3://...)")
    s3_bucket = parsed.netloc
    s3_folder = os.path.dirname(parsed.path).lstrip("/")
    local_filename = "user_keypoints.npy"
    np.save(local_filename, user_keypoints)
    s3_key = os.path.join(s3_folder, local_filename)
    return upload_to_s3(local_filename, s3_bucket, s3_key, aws_access_key_id, aws_secret_access_key, region_name)

if __name__ == "__main__":
    # Example usage
    dummy_keypoints = np.random.rand(100, 17, 3)
    output_video_s3_url = "s3://your-s3-bucket-name/your/output/folder/output_video.mp4"
    save_and_upload_user_keypoints(dummy_keypoints, output_video_s3_url)
