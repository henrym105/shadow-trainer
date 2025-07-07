import os
import logging
import boto3
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class S3Manager:
    def __init__(self):
        self.s3_client = boto3.client('s3')
    
    def is_s3_path(self, path: str) -> bool:
        """Check if the given path is an S3 path."""
        return path.startswith("s3://")
    
    def download_from_s3(self, s3_path: str, local_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Download a file from S3 to local path.
        
        Args:
            s3_path (str): S3 path in format s3://bucket/key
            local_path (str): Local file path to save the downloaded file
            
        Returns:
            Tuple[bucket, key, error]: bucket and key if successful, error message if failed
        """
        s3_path_no_prefix = s3_path[5:]  # Remove 's3://' prefix
        bucket, key = s3_path_no_prefix.split('/', 1)
        
        try:
            logger.info(f"Downloading {s3_path} to {local_path}")
            self.s3_client.download_file(bucket, key, local_path)
            return bucket, key, None
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return None, None, str(e)
    
    def upload_to_s3(self, local_path: str, bucket: str, video_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Upload a file to S3.
        
        Args:
            local_path (str): Path to the local file to upload
            bucket (str): S3 bucket name
            video_name (str): Name of the video (used in S3 key)
            
        Returns:
            Tuple[s3_url, error]: S3 URL if successful, error message if failed
        """
        output_video_s3_key = f"tmp/{video_name}/{os.path.basename(local_path)}"
        
        try:
            logger.info(f"Uploading output video {local_path} to s3://{bucket}/{output_video_s3_key}")
            self.s3_client.upload_file(local_path, bucket, output_video_s3_key)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{output_video_s3_key}")
            output_video_s3_url = f"s3://{bucket}/{output_video_s3_key}"
            return output_video_s3_url, None
        except Exception as e:
            logger.error(f"Failed to upload output video to S3: {e}")
            return None, str(e)
