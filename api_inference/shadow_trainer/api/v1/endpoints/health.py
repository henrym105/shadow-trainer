"""
Health check endpoints for Shadow Trainer API.
"""
import logging
import time
import psutil
from fastapi import APIRouter, Depends
from typing import Dict, Any

from shadow_trainer.config import settings
from shadow_trainer.schemas.responses import HealthResponse
from shadow_trainer.services.model_service import ModelService
from shadow_trainer.api.dependencies import model_service_dependency

logger = logging.getLogger(__name__)
router = APIRouter()

# Track application start time
_start_time = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    uptime = time.time() - _start_time
    
    return HealthResponse(
        success=True,
        status="healthy",
        version=settings.version,
        uptime_seconds=uptime,
        message=f"{settings.app_name} is running"
    )

@router.get("/health/detailed", response_model=HealthResponse)
async def detailed_health_check(
    model_service: ModelService = Depends(model_service_dependency)
):
    """Detailed health check with system and model information."""
    try:
        uptime = time.time() - _start_time
        
        # Get system information
        system_info = _get_system_info()
        
        # Get model status
        model_status_result = model_service.get_model_status()
        model_status = model_status_result.get("model_status", {})
        
        return HealthResponse(
            success=True,
            status="healthy",
            version=settings.version,
            uptime_seconds=uptime,
            model_status=model_status,
            system_info=system_info,
            message=f"{settings.app_name} is running with detailed status"
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return HealthResponse(
            success=False,
            status="degraded",
            version=settings.version,
            uptime_seconds=time.time() - _start_time,
            message=f"Health check partially failed: {str(e)}"
        )

@router.get("/ping")
async def ping():
    """Simple ping endpoint for load balancers."""
    return {"status": "ok", "timestamp": time.time()}

def _get_system_info() -> Dict[str, Any]:
    """Get system information for health check."""
    try:
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent_used": memory.percent
        }
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_info = {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": round((disk.used / disk.total) * 100, 2)
        }
        
        return {
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "memory": memory_info,
            "disk": disk_info,
            "python_version": f"{psutil.version_info}",
        }
        
    except Exception as e:
        logger.warning(f"Could not get system info: {e}")
        return {"error": "System info unavailable"}
