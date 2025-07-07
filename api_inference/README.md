# API Service Architecture

This directory contains the refactored API service code for the MotionAGFormer video processing application. The code has been organized into multiple modules for better maintainability and separation of concerns.

## File Structure

### Core Files

- **`api_service.py`** - Main FastAPI application with route definitions
- **`endpoints.py`** - API endpoint logic and request handling
- **`pipeline.py`** - Video processing pipeline operations
- **`models.py`** - Model configuration management
- **`s3_utils.py`** - S3 upload/download utilities
- **`file_utils.py`** - File validation and management utilities
- **`config.py`** - Configuration constants and settings

### Supporting Files

- **`inference.py`** - Core inference functions (existing)
- **`model_config_map.json`** - Model configuration mapping (existing)
- **`utils.py`** - General utilities (existing)

## Module Responsibilities

### `config.py`
- Application configuration constants
- Directory paths
- Supported file extensions
- Default S3 bucket configuration

### `s3_utils.py`
- S3Manager class for S3 operations
- File upload/download functionality
- S3 path validation

### `file_utils.py`
- File extension validation
- Directory cleanup utilities
- Directory creation helpers

### `models.py`
- Model configuration loading
- Model size mapping
- Configuration file parsing

### `pipeline.py`
- Video processing pipeline orchestration
- Calls to pose estimation and video generation functions
- Error handling for pipeline operations

### `endpoints.py`
- Individual endpoint implementations
- Request validation and processing
- Response formatting
- Error handling

### `api_service.py`
- FastAPI application initialization
- Route definitions
- Application entry point

## Benefits of This Structure

1. **Separation of Concerns** - Each module has a specific responsibility
2. **Maintainability** - Easier to modify individual components
3. **Testability** - Modules can be tested independently
4. **Reusability** - Utilities can be reused across different parts of the application
5. **Readability** - Smaller, focused files are easier to understand

## Usage

The main application is still started through `api_service.py`:

```bash
python api_service.py
```

Or with uvicorn directly:

```bash
uvicorn api_service:app --host 0.0.0.0 --port 8000
```

All existing API endpoints remain unchanged, ensuring backward compatibility.
