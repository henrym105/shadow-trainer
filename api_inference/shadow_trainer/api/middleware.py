"""
Custom middleware for Shadow Trainer API.
"""
import time
import logging
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    """Log all incoming requests and their processing time."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        f"Response: {response.status_code} | "
        f"Time: {process_time:.3f}s | "
        f"Method: {request.method} | "
        f"Path: {request.url.path}"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """Add security headers to responses."""
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response

def add_custom_middleware(app: FastAPI) -> None:
    """Add all custom middleware to the FastAPI application."""
    
    # Add request logging middleware
    app.middleware("http")(request_logging_middleware)
    
    # Add security headers middleware
    app.middleware("http")(security_headers_middleware)
    
    logger.info("Custom middleware added to application")
