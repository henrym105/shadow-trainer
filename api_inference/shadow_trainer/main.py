"""
FastAPI application factory for Shadow Trainer.
"""
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from shadow_trainer.config import settings, ensure_directories
from shadow_trainer.api.v1.router import api_router
from shadow_trainer.api.middleware import add_custom_middleware
from shadow_trainer.frontend.routes import frontend_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format
)

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    # Ensure directories exist
    ensure_directories()
    
    # Create FastAPI instance
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="FastAPI application for 3D human pose estimation from video",
        debug=settings.debug,
        docs_url=settings.docs_url if settings.debug else None,
        redoc_url=settings.redoc_url if settings.debug else None,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    add_custom_middleware(app)
    
    # Mount static files for frontend
    frontend_static_dir = Path(__file__).parent / "frontend" / "static"
    if frontend_static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_static_dir)), name="static")
        logger.info(f"Mounted frontend static files from {frontend_static_dir}")
    
    # Mount static files for assets
    if settings.assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(settings.assets_dir)), name="assets")
        logger.info(f"Mounted static assets from {settings.assets_dir}")
    
    # Include frontend router
    app.include_router(frontend_router, prefix="/frontend", tags=["Frontend"])
    
    # Include API router
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    
    # Root endpoint serves the frontend UI
    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """Root endpoint that serves the main frontend page."""
        # Get template directory
        templates_dir = Path(__file__).parent / "frontend" / "templates"
        templates = Jinja2Templates(directory=str(templates_dir))
        
        return templates.TemplateResponse("index.html", {"request": request})
    
    # Alternative UI endpoint for backward compatibility
    @app.get("/ui", response_class=HTMLResponse)
    async def ui_redirect(request: Request):
        """Alternative UI endpoint that serves the main frontend page."""
        # Get template directory
        templates_dir = Path(__file__).parent / "frontend" / "templates"
        templates = Jinja2Templates(directory=str(templates_dir))
        
        return templates.TemplateResponse("index.html", {"request": request})
    
    # Health check endpoint
    @app.get("/health")
    async def root_health():
        return {"status": "healthy", "version": settings.version}
    
    # API info endpoint
    @app.get("/info")
    async def api_info():
        return {
            "message": f"{settings.app_name} is running",
            "version": settings.version,
            "docs_url": f"{settings.api_v1_prefix}/docs" if settings.debug else None
        }
    
    logger.info(f"FastAPI application created: {settings.app_name} v{settings.version}")
    return app

# Create application instance
app = create_app()
