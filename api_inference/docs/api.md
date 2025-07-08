# Shadow Trainer API Documentation

## Overview

The Shadow Trainer API is a comprehensive FastAPI application for 3D human pose estimation from video. This document provides detailed information about the API endpoints, architecture, and usage.

## Architecture

The application follows a clean architecture pattern with clear separation of concerns:

- **API Layer**: FastAPI endpoints and request/response handling
- **Service Layer**: Business logic and orchestration
- **Core Layer**: Inference engine, storage, and utilities
- **Storage Layer**: Abstraction for local and S3 storage operations

## API Endpoints

### Base URL
- Development: `http://localhost:8000`
- API Base: `/api/v1`

### Health Endpoints

#### GET `/health`
Basic health check.

**Response:**
```json
{
  "success": true,
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "message": "Shadow Trainer API is running"
}
```

#### GET `/health/detailed`
Detailed health check with system information.

**Response:**
```json
{
  "success": true,
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "model_status": {
    "xs": {"available": true, "config_path": "/path/to/config"},
    "s": {"available": true, "config_path": "/path/to/config"},
    "b": {"available": true, "config_path": "/path/to/config"},
    "l": {"available": true, "config_path": "/path/to/config"}
  },
  "system_info": {
    "cpu_percent": 25.5,
    "cpu_count": 8,
    "memory": {
      "total_gb": 16.0,
      "available_gb": 12.5,
      "percent_used": 21.9
    },
    "disk": {
      "total_gb": 500.0,
      "free_gb": 350.0,
      "percent_used": 30.0
    }
  }
}
```

### Video Processing Endpoints

#### POST `/api/v1/video/process`
Process a video from a file path (local or S3).

**Request Body:**
```json
{
  "file_path": "s3://bucket/path/to/video.mp4",
  "model_size": "xs",
  "handedness": "Right-handed", 
  "pitch_types": ["FF", "SL"]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Video processed successfully",
  "output_video_local_path": "/path/to/output.mp4",
  "output_video_s3_url": "s3://bucket/processed/output.mp4",
  "processing_time_seconds": 45.2,
  "model_size_used": "xs",
  "metadata": {
    "job_id": "uuid",
    "input_path": "s3://bucket/path/to/video.mp4",
    "handedness": "Right-handed",
    "pitch_types": ["FF", "SL"]
  }
}
```

#### POST `/api/v1/video/upload-and-process`
Upload and process a video file in one request.

**Form Data:**
- `video_file`: Video file (multipart/form-data)
- `model_size`: Model size (xs, s, b, l) - optional, default: xs
- `handedness`: Handedness (Right-handed, Left-handed) - optional, default: Right-handed
- `pitch_types`: Comma-separated pitch types - optional

**Response:** Same as `/process` endpoint

#### GET `/api/v1/video/sample-videos`
Get list of available sample videos.

**Response:**
```json
{
  "success": true,
  "videos": [
    {
      "name": "sample1.mp4",
      "size": 15728640,
      "url": "/assets/videos/sample1.mp4",
      "duration_seconds": 30.0
    }
  ],
  "count": 1
}
```

#### POST `/api/v1/video/cleanup`
Clean up old temporary files.

**Query Parameters:**
- `max_age_hours`: Maximum age of files to keep (default: 24)

**Response:**
```json
{
  "success": true,
  "cleaned_files": 5,
  "message": "Cleaned up 5 old files"
}
```

#### GET `/api/v1/video/supported-formats`
Get information about supported formats and limits.

**Response:**
```json
{
  "supported_extensions": [".mp4", ".mov", ".avi"],
  "max_file_size_bytes": 104857600,
  "max_file_size_formatted": "100.0 MB",
  "max_processing_time_seconds": 300,
  "supported_model_sizes": ["xs", "s", "b", "l"],
  "default_model_size": "xs"
}
```

### Model Management Endpoints

#### GET `/api/v1/models/list`
List all available models.

**Response:**
```json
{
  "success": true,
  "models": {
    "xs": {
      "model_size": "xs",
      "available": true,
      "config_path": "/path/to/config",
      "description": "Extra small model for fast processing"
    }
  },
  "supported_sizes": ["xs", "s", "b", "l"],
  "default_size": "xs"
}
```

#### GET `/api/v1/models/config/{model_size}`
Get configuration for a specific model.

**Response:**
```json
{
  "success": true,
  "model_size": "xs",
  "config_path": "/path/to/config.yaml",
  "checkpoint_path": "/path/to/checkpoint.pth",
  "parameters": {
    "input_size": [256, 256],
    "num_joints": 17
  }
}
```

#### GET `/api/v1/models/status`
Get status of all models.

**Response:**
```json
{
  "success": true,
  "model_status": {
    "xs": {"available": true, "config_exists": true},
    "s": {"available": true, "config_exists": true},
    "b": {"available": false, "error": "Config file not found"},
    "l": {"available": true, "config_exists": true}
  },
  "total_models": 4,
  "available_models": 3
}
```

## Error Handling

All endpoints return standardized error responses:

```json
{
  "success": false,
  "message": "Error description",
  "error_code": "ERROR_CODE", 
  "details": {
    "additional": "error details"
  }
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (file/resource not found)
- `500`: Internal Server Error

## Configuration

The application can be configured via environment variables with the `SHADOW_TRAINER_` prefix:

- `SHADOW_TRAINER_DEBUG`: Enable debug mode (default: false)
- `SHADOW_TRAINER_HOST`: Server host (default: 0.0.0.0)
- `SHADOW_TRAINER_PORT`: Server port (default: 8000)
- `SHADOW_TRAINER_S3_BUCKET`: S3 bucket for storage (default: shadow-trainer-prod)
- `SHADOW_TRAINER_MAX_FILE_SIZE`: Maximum file size in bytes (default: 104857600)

## Model Sizes

- `xs`: Extra Small - Fastest processing, lower accuracy
- `s`: Small - Good balance of speed and accuracy
- `b`: Base - Higher accuracy, moderate speed  
- `l`: Large - Highest accuracy, slower processing

## Supported Video Formats

- `.mp4`: H.264/H.265 encoded videos
- `.mov`: QuickTime videos
- `.avi`: Audio Video Interleave files

Maximum file size: 100MB  
Maximum processing time: 5 minutes
