
# Shadow Trainer - AI-Powered Motion Analysis

Shadow Trainer is a professional video processing platform for athletic motion analysis, powered by YOLO pose estimation and deep learning. It provides real-time feedback and pose comparison against professional athletes.


## Architecture

**Monolithic deployment** on AWS EC2 with nginx reverse proxy:
- **Frontend**: React.js app (`api_frontend/`)
- **Backend**: FastAPI Python service (`api_backend/`)
- **Proxy**: Nginx for routing and SSL
- **Storage**: Local filesystem


## Quick Start

### Prerequisites
- Linux EC2 instance (Amazon Linux 2023 recommended)
- Python 3.9+ (with [uv](https://github.com/astral-sh/uv) package manager)
- Node.js 18+ and npm
- Nginx

### Setup & Running (via Makefile)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/henrym105/shadow-trainer.git
   cd shadow-trainer
   ```

2. **Install Python and Node dependencies:**
   ```bash
   ./scripts/install_setup_uv.sh
   cd api_frontend && npm install
   ```


4. **Start services (dev or prod):**
   ```bash
   make dev        # Start development stack
   make prod       # Start production stack
   ```

5. **View logs:**
   ```bash
   make dev-logs   # Development logs
   make prod-logs  # Production logs
   ```

6. **Stop all services:**
   ```bash
   make stop
   ```

7. **Health check:**
   ```bash
   make health
   ```


## API Endpoints

### Core Video Processing
- `POST /api/videos/upload` — Upload video and start processing
- `GET /api/videos/{job_id}/status` — Check processing status
- `GET /api/videos/{job_id}/download` — Download processed video
- `GET /api/videos/{job_id}/preview` — Stream video for preview

### System
- `GET /health` — Health check endpoint


## Frontend Features

- **Drag & Drop Upload** — Intuitive file upload with validation
- **Real-time Progress** — Live updates during video processing
- **Model Selection** — Choose analysis quality/speed
- **Video Preview** — Stream and download processed videos
- **Error Handling** — User-friendly error messages and recovery
- **Professional Comparison** — Compare your motion to pro athletes
- **Configurable Output** — Choose 2D+3D or 3D-only video


## Backend Features

- **Async Processing** — Background job queue with progress updates
- **File Validation** — Type and size checks for uploads
- **Progress Tracking** — Real-time updates during processing
- **Error Recovery** — Graceful error handling and cleanup
- **Health Monitoring** — System health checks
- **Model Inference** — YOLO 2D pose + MotionAGFormer 3D pose
- **Professional Keypoints** — Compare user and pro motion


## Development

### Project Structure & Task Processing
```
shadow-trainer/
├── api_backend/           # FastAPI backend service
│   ├── api_service.py     # Main API endpoints (FastAPI)
│   ├── tasks.py           # Celery task definitions for async job processing
│   ├── pydantic_models.py # Data models for API
│   ├── src/               # ML pipeline (YOLO, MotionAGFormer, etc.)
│   └── ...
├── api_frontend/          # React frontend application
│   ├── src/               # Components, services, main app
│   └── public/            # Static assets
├── scripts/               # Setup and deployment scripts
├── Makefile               # Unified dev/prod commands
└── nginx/                 # Nginx configuration
```

#### Task Processing Flow (Celery)

1. **User uploads video via frontend** → FastAPI receives upload
2. **FastAPI validates and saves file**
3. **Celery task is triggered** (`process_video_task` in `tasks.py`)
4. **Task runs in background worker** (Redis broker)
   - Video is processed: upright correction, 2D/3D pose estimation, pro comparison, frame generation
   - Progress/status is tracked and updated
5. **Frontend polls for status** via `/videos/{task_id}/status`
6. **User downloads or previews result** when complete

This decoupled architecture allows scalable, non-blocking video processing and real-time progress updates.


### Adding New Features

**Backend:**
- Add endpoints in `api_service.py`
- Update models in `pydantic_models.py`
- Extend job logic in `job_manager.py`

**Frontend:**
- Add API calls in `src/services/videoApi.js`
- Create new components in `src/components/`
- Update main app logic in `src/App.js`

**Infrastructure:**
- Update nginx config in `scripts/create_nginx_config.sh`
- Modify Makefile/scripts as needed


## Model Configuration

Supports multiple model sizes for quality/speed tradeoffs:
- `xs` — Extra Small (fastest, lowest quality)
- `s` — Small (balanced)
- `m` — Medium (higher quality)
- `l` — Large (highest quality, slowest)

Configuration: `api_backend/model_config_map.json`


## Error Handling

- **File Validation** — Type and size checks before upload
- **Network Errors** — Automatic retry with user feedback
- **Processing Errors** — Detailed error messages and recovery
- **Polling Failures** — Graceful fallback for real-time updates


## Deployment Notes

- Uses Docker Compose for dev/prod orchestration
- Makefile for unified commands
- File cleanup runs automatically during health checks
- 100MB max file upload size
- Real-time progress updates every 2 seconds


## Security

- CORS configured for development (update for production)
- File type validation prevents malicious uploads
- Size limits prevent resource exhaustion
- Security headers in nginx
- No authentication yet (planned)


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
