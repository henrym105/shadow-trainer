"""
S3 storage operations for Shadow Trainer.
"""
import logging
import boto3
from pathlib import Path
from typing import Optional, Tuple
from botocore.exceptions import ClientError, NoCredentialsError

from shadow_trainer.config import settings

logger = logging.getLogger(__name__)

class S3Storage:
    """S3 storage manager for file operations."""
    
    def __init__(self):
        """Initialize S3 client."""
        self.bucket = settings.s3_bucket
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize S3 client with credentials."""
        try:
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                self.client = boto3.client(
                    's3',
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                    region_name=settings.aws_region
                )
            else:
                # Use default credential chain (IAM roles, env vars, etc.)
                self.client = boto3.client('s3', region_name=settings.aws_region)
            
            logger.info(f"S3 client initialized for bucket: {self.bucket}")
            
        except NoCredentialsError:
            logger.warning("No AWS credentials found. S3 operations will not be available.")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if S3 client is available."""
        return self.client is not None
    
    def is_s3_path(self, path: str) -> bool:
        """Check if a path is an S3 URL."""
        return path.startswith('s3://')
    
    def parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        """
        Parse S3 path into bucket and key.
        
        Args:
            s3_path: S3 URL (s3://bucket/key)
            
        Returns:
            Tuple of (bucket, key)
        """
        if not s3_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path: {s3_path}")
        
        path_parts = s3_path[5:].split('/', 1)
        if len(path_parts) != 2:
            raise ValueError(f"Invalid S3 path format: {s3_path}")
        
        return path_parts[0], path_parts[1]
    
    async def download_file(self, s3_path: str, local_dir: Path) -> Path:
        """
        Download a file from S3 to local directory.
        
        Args:
            s3_path: S3 URL to download from
            local_dir: Local directory to save file
            
        Returns:
            Path to downloaded local file
        """
        if not self.is_available():
            raise RuntimeError("S3 client not available")
        
        bucket, key = self.parse_s3_path(s3_path)
        filename = Path(key).name
        local_path = local_dir / filename
        
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {s3_path} to {local_path}")
            self.client.download_file(bucket, key, str(local_path))
            
            logger.info(f"Successfully downloaded {s3_path}")
            return local_path
            
        except ClientError as e:
            error_msg = f"Failed to download {s3_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def upload_file(
        self, 
        local_path: Path, 
        s3_key: Optional[str] = None,
        bucket: Optional[str] = None
    ) -> str:
        """
        Upload a local file to S3.
        
        Args:
            local_path: Path to local file
            s3_key: S3 key (if None, uses filename)
            bucket: S3 bucket (if None, uses default)
            
        Returns:
            S3 URL of uploaded file
        """
        if not self.is_available():
            raise RuntimeError("S3 client not available")
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        bucket = bucket or self.bucket
        s3_key = s3_key or local_path.name
        
        try:
            logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
            
            self.client.upload_file(
                str(local_path), 
                bucket, 
                s3_key,
                ExtraArgs={'ContentType': self._get_content_type(local_path)}
            )
            
            s3_url = f"s3://{bucket}/{s3_key}"
            logger.info(f"Successfully uploaded to {s3_url}")
            return s3_url
            
        except ClientError as e:
            error_msg = f"Failed to upload {local_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _get_content_type(self, file_path: Path) -> str:
        """Get content type for file upload."""
        suffix = file_path.suffix.lower()
        content_types = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        return content_types.get(suffix, 'application/octet-stream')
    
    def download_from_s3(self, s3_path: str, local_path: str) -> Tuple[str, str, Optional[str]]:
        """
        Legacy function for backward compatibility.
        
        Returns:
            Tuple of (bucket, key, error_message)
        """
        try:
            bucket, key = self.parse_s3_path(s3_path)
            local_dir = Path(local_path).parent
            local_dir.mkdir(parents=True, exist_ok=True)
            
            if not self.is_available():
                return bucket, key, "S3 client not available"
            
            self.client.download_file(bucket, key, local_path)
            return bucket, key, None
            
        except Exception as e:
            return "", "", str(e)
    
    def upload_to_s3(self, local_path: str, bucket: str, video_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Legacy function for backward compatibility.
        
        Returns:
            Tuple of (s3_url, error_message)
        """
        try:
            if not self.is_available():
                return None, "S3 client not available"
            
            local_file = Path(local_path)
            if not local_file.exists():
                return None, f"Local file not found: {local_path}"
            
            s3_key = f"processed_videos/{video_name}_{local_file.name}"
            s3_url = f"s3://{bucket}/{s3_key}"
            
            self.client.upload_file(local_path, bucket, s3_key)
            return s3_url, None
            
        except Exception as e:
            return None, str(e)
