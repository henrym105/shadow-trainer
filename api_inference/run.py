#!/usr/bin/env python3
"""
Application entry point for Shadow Trainer API.
"""
import sys
from pathlib import Path

# Add the api_inference directory to Python path
API_INFERENCE_DIR = Path(__file__).parent
sys.path.insert(0, str(API_INFERENCE_DIR))

if __name__ == "__main__":
    import uvicorn
    from shadow_trainer.config import settings
    
    # Run the application
    uvicorn.run(
        "shadow_trainer.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.log_level.lower()
    )
