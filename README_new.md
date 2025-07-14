# Shadow Trainer - AI-Powered Motion Analysis

A professional video processing application that uses YOLO pose estimation to analyze athletic performance and provide motion feedback.

## Architecture

**Monolith deployment** on AWS EC2 with nginx reverse proxy:
- **Frontend**: React.js application (served on port 8000)
- **Backend**: FastAPI Python service (served on port 8002)  
- **Proxy**: Nginx routes traffic and handles SSL termination
- **Storage**: Local file system (will expand to S3 in future)

## Quick Start

### Prerequisites
- Linux EC2 instance (Amazon Linux 2023 recommended)
- Python 3.9+ with UV package manager
- Node.js 18+ with npm
- Nginx

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository>
cd shadow-trainer
./scripts/install_setup_uv.sh
```

2. **Install frontend dependencies:**
```bash
cd api_frontend/shadow_trainer_web
npm install
```

3. **Configure nginx:**
```bash
./scripts/create_nginx_config.sh
```

4. **Start all services:**
```bash
./start_all.sh
```

5. **Stop all services:**
```bash
./stop_all.sh
```

## API Endpoints

### Core Video Processing
- `POST /api/videos/upload` - Upload video and start processing
- `GET /api/videos/{job_id}/status` - Check processing status  
- `GET /api/videos/{job_id}/download` - Download processed video
- `GET /api/videos/{job_id}/preview` - Stream video for preview

### System
- `GET /health` - Health check endpoint

## Frontend Features

- **Drag & Drop Upload**: Intuitive file upload with validation
- **Real-time Progress**: Live updates during video processing
- **Model Selection**: Choose from different analysis quality levels
- **Video Preview**: Stream and download processed videos
- **Error Handling**: User-friendly error messages and recovery

## Backend Features

- **Async Processing**: Background job processing with progress updates
- **File Validation**: Type and size validation for uploads
- **Progress Tracking**: Real-time progress updates during processing
- **Error Recovery**: Graceful error handling and cleanup
- **Health Monitoring**: System health checks and monitoring

## Development

### Project Structure
```
shadow-trainer/
├── api_backend/           # FastAPI backend service
│   ├── api_service.py     # Main API endpoints
│   ├── job_manager.py     # Job queue and processing
│   ├── pydantic_models.py # Data models
│   └── src/               # ML processing pipeline
├── api_frontend/          # React frontend application  
│   └── shadow_trainer_web/
│       ├── src/
│       │   ├── components/    # React components
│       │   ├── services/      # API client code
│       │   └── App.js         # Main application
│       └── public/            # Static assets
├── scripts/               # Setup and deployment scripts
└── nginx/                 # Nginx configuration
```

### Adding New Features

1. **Backend API Changes:**
   - Add new endpoints in `api_service.py`
   - Update models in `pydantic_models.py`
   - Extend job processing in `job_manager.py`

2. **Frontend Changes:**
   - Add API calls in `services/videoApi.js`
   - Create new components in `components/`
   - Update main app logic in `App.js`

3. **Infrastructure:**
   - Update nginx config in `scripts/create_nginx_config.sh`
   - Modify startup scripts as needed

## Model Configuration

The system supports multiple model sizes for different quality/speed tradeoffs:
- `xs` - Extra Small (fastest, lowest quality)
- `s` - Small (balanced)
- `m` - Medium (higher quality)
- `l` - Large (highest quality, slowest)

Configuration is managed in `api_backend/model_config_map.json`.

## Error Handling

- **File Validation**: Type and size checks before upload
- **Network Errors**: Automatic retry with user feedback  
- **Processing Errors**: Detailed error messages and recovery
- **Polling Failures**: Graceful degradation of real-time updates

## Deployment Notes

- Uses tmux for process management during development
- Production should use systemd services
- File cleanup runs automatically during health checks
- Supports 100MB maximum file upload size
- Real-time progress updates every 2 seconds

## Security

- CORS configured for development (update for production)
- File type validation prevents malicious uploads
- Size limits prevent resource exhaustion
- Security headers configured in nginx
- No authentication currently (add as needed)

## Performance

- Background processing prevents UI blocking
- File streaming for large video downloads
- Nginx compression for static assets
- Progress caching for better UX
- Cleanup jobs prevent disk space issues

## Future Enhancements

- [ ] AWS S3 integration for scalable storage
- [ ] Database persistence for job history  
- [ ] User authentication and accounts
- [ ] Multiple concurrent video processing
- [ ] Real-time WebSocket updates
- [ ] Advanced analytics and reporting
- [ ] Mobile app support
