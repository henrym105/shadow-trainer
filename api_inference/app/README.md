# Shadow Trainer FastAPI Application

This directory contains the restructured Shadow Trainer application that combines both frontend UI and backend API endpoints in a single FastAPI application.

## Structure

```
api_inference/
├── app/                          # Main application package
│   ├── main.py                   # FastAPI application entry point
│   ├── api/                      # Backend API routes
│   │   ├── __init__.py
│   │   └── routes.py             # API endpoints for video processing
│   ├── frontend/                 # Frontend UI components
│   │   ├── __init__.py
│   │   ├── routes.py             # Frontend routes and upload handling
│   │   ├── templates/            # Jinja2 HTML templates
│   │   │   └── index.html        # Main UI page
│   │   └── static/               # Static assets
│   │       ├── css/
│   │       │   └── style.css     # Custom styles
│   │       ├── js/
│   │       │   └── app.js        # Frontend JavaScript
│   │       ├── images/
│   │       │   └── logo.png      # Shadow Trainer logo
│   │       └── videos/           # Sample videos
│   └── core/                     # Core configuration and utilities
│       ├── __init__.py
│       └── config.py             # Application configuration
├── run_app.py                    # Application startup script
└── [existing files...]          # Original API inference modules
```

## Features

### Frontend UI
- **Modern Web Interface**: Clean, responsive UI built with Bootstrap 5
- **Video Upload**: Support for local file uploads and S3 paths
- **Configuration Options**: Model size selection, handedness, and pitch type preferences
- **Real-time Processing**: Live status updates and result display
- **Sample Videos**: Preview of available sample videos

### Backend API
- **RESTful Endpoints**: `/api/v1/process_video/`, `/api/v1/health`, etc.
- **File Upload Support**: Handle multipart form data for video uploads
- **S3 Integration**: Process videos directly from S3 paths
- **Error Handling**: Comprehensive error responses and logging

## Running the Application

### Quick Start
```bash
# From the api_inference directory
python run_app.py
```

### Development Mode
```bash
# With auto-reload for development
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access Points
- **Frontend UI**: http://localhost:8000/ or http://localhost:8000/ui
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **API Base**: http://localhost:8000/api/v1/

## API Endpoints

### Video Processing
- `POST /api/v1/process_video/` - Process a video file
  - Parameters: `file`, `model_size`, `handedness`, `pitch_type`

### Health Checks
- `GET /api/v1/health` - Application health status
- `GET /api/v1/ping` - SageMaker-compatible health check

### Frontend
- `GET /` - Main UI page
- `GET /ui` - Alternative UI access point
- `POST /upload_video` - Handle file uploads from frontend

## Dependencies

The application requires the following additional packages (added to pyproject.toml):
- `fastapi>=0.104.1` - Web framework
- `uvicorn[standard]>=0.23.2` - ASGI server
- `jinja2>=3.1.2` - Template engine
- `python-multipart>=0.0.6` - File upload support
- `aiofiles>=23.2.1` - Async file operations

## Migration from Streamlit

This new structure replaces the previous Streamlit application while maintaining all functionality:

1. **UI Features**: All Streamlit features are preserved in the new web interface
2. **API Compatibility**: Existing API endpoints remain functional
3. **File Handling**: Improved file upload and processing capabilities
4. **Performance**: Better scalability and production readiness

## Development

### Adding New Features
1. **API Endpoints**: Add new routes in `app/api/routes.py`
2. **Frontend Pages**: Create new templates and update `app/frontend/routes.py`
3. **Static Assets**: Add CSS, JS, or images to `app/frontend/static/`

### Configuration
- Modify `app/core/config.py` for application settings
- Environment variables can be used for deployment-specific configuration
