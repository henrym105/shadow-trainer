"""
API v1 router aggregation for Shadow Trainer.
"""
from fastapi import APIRouter

from .endpoints.health import router as health_router
from .endpoints.video import router as video_router
from .endpoints.models import router as models_router

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    health_router,
    prefix="/health",
    tags=["Health"]
)

api_router.include_router(
    video_router,
    prefix="/video",
    tags=["Video Processing"]
)

api_router.include_router(
    models_router,
    prefix="/models", 
    tags=["Model Management"]
)

# Legacy endpoints for backward compatibility
@api_router.post("/process_video/")
async def legacy_process_video(
    file: str,
    model_size: str = "xs",
    handedness: str = "Right-handed",
    pitch_type: str = ""
):
    """Legacy endpoint for backward compatibility."""
    from shadow_trainer.schemas.requests import VideoProcessRequest
    from shadow_trainer.schemas.common import ModelSize, Handedness, PitchType
    from shadow_trainer.api.dependencies import video_service_dependency
    
    # Convert legacy parameters to new format
    try:
        model_size_enum = ModelSize(model_size)
    except ValueError:
        model_size_enum = ModelSize.XS
    
    try:
        handedness_enum = Handedness(handedness)
    except ValueError:
        handedness_enum = Handedness.RIGHT
    
    # Parse pitch types
    pitch_types = []
    if pitch_type:
        for pt in pitch_type.split(","):
            pt = pt.strip().upper()
            try:
                pitch_types.append(PitchType(pt))
            except ValueError:
                continue
    
    # Create request
    request = VideoProcessRequest(
        file_path=file,
        model_size=model_size_enum,
        handedness=handedness_enum,
        pitch_types=pitch_types
    )
    
    # Process using video service
    video_service = await video_service_dependency()
    result = await video_service.process_video(request)
    
    # Return in legacy format
    if result.success:
        return {
            "output_video_local_path": result.output_video_local_path,
            "output_video_s3_url": result.output_video_s3_url
        }
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=result.message)

@api_router.get("/")
async def legacy_root():
    """Legacy root endpoint."""
    return {"message": "MotionAGFormer API is running."}

@api_router.get("/ping")
async def legacy_ping():
    """Legacy ping endpoint."""
    from fastapi.responses import Response
    return Response(status_code=200)

@api_router.post("/invocations")
async def legacy_invocations(request):
    """Legacy SageMaker invocations endpoint."""
    # This would need to be implemented based on the original invocations logic
    from fastapi import HTTPException
    raise HTTPException(status_code=501, detail="Legacy invocations endpoint not yet implemented")
