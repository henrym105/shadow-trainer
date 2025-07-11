from pydantic import BaseModel
from typing import Optional


class ProcessVideoRequest(BaseModel):
    file: str
    model_size: str = "xs"


class ProcessVideoResponse(BaseModel):
    output_video_local_path: str
    output_video_s3_url: Optional[str] = None
    error: Optional[str] = None






