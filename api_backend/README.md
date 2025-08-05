# Shadow Trainer Backend

Python FastAPI backend service providing video processing, machine learning inference, and biomechanical analysis APIs for the Shadow Trainer motion analysis platform.

## 🏗️ Backend Architecture

### Core Stack
- **FastAPI 0.116+** - Modern Python web framework with async support
- **Celery 5.5+** - Distributed task queue with Redis broker  
- **Redis 6.0+** - In-memory data store for task queuing and caching
- **PyTorch 2.3+** - Deep learning framework for ML models
- **OpenCV 4.11+** - Computer vision and video processing

### Machine Learning Pipeline
- **YOLO11x-pose** - State-of-the-art 2D human pose estimation
- **MotionAGFormer** - 3D pose estimation from 2D keypoints
- **Custom Motion Analysis** - Baseball-specific biomechanical calculations
- **OpenAI API** - LLM-powered coaching feedback generation

## 📁 Project Structure

```
api_backend/
├── api_service.py          # FastAPI app initialization and middleware
├── src/                    # Core processing modules
│   ├── inference.py        # ML model inference pipeline
│   ├── yolo2d.py           # 2D pose detection with YOLO11x
│   ├── movement_analysis.py # Biomechanical analysis algorithms
│   ├── visualizations.py   # 3D rendering and chart generation
│   ├── preprocess.py       # Video preprocessing utilities
│   ├── utils.py            # Common utility functions
│   └── model/              # MotionAGFormer implementation
│       ├── MotionAGFormer.py
│       └── modules/        # Neural network components
├── checkpoint/             # Pre-trained model weights loaded from s3
├── output/                # Processed video results storage
├── uploads/               # Temporary video upload storage
└── sample_videos/         # Test videos and sample outputs
```

## 🔌 API Endpoints

> **Full API Documentation:**  
> See [api.shadow-trainer.com/docs](https://api.shadow-trainer.com/docs) for complete, interactive API reference and usage details.

### Video Processing
```bash
POST /videos/upload/                      # Upload video and start processing
GET  /videos/{task_id}/info/              # Get task metadata info dict
GET  /videos/{task_id}/status/            # Check processing status
GET  /videos/{task_id}/download/          # Download processed video
GET  /videos/{task_id}/preview/           # Stream video preview, original or processed
POST /videos/{task_id}/terminate/         # Cancel processing job
```

### 3D Data & Analysis
```bash
GET  /videos/{task_id}/keypoints/user/    # User 3D keypoints (JSON)
GET  /videos/{task_id}/keypoints/pro/     # Professional athlete keypoints
GET  /videos/{task_id}/analysis-plots/    # Movement analysis charts
POST /videos/{task_id}/generate-evaluation/ # Generate AI coaching feedback
```

### System Management
```bash
GET  /health/                             # System health check
GET  /pro-keypoints/list                  # Available professional athletes
GET  /status/{task_id}
```

## 🔄 Processing Pipeline

1. **Video Upload** → FastAPI validates and stores file
2. **Task Queue** → Celery worker picks up processing job  
3. **2D Pose Detection** → YOLO11x extracts joint coordinates
4. **3D Pose Estimation** → MotionAGFormer converts to 3D keypoints
5. **Biomechanical Analysis** → Calculate joint angles and velocities
6. **AI Feedback** → OpenAI generates coaching insights
7. **Results Storage** → Save outputs with metadata

## 🛠️ Development

### Local Setup
```bash
# Install dependencies with UV package manager
cd api_backend
uv pip install -r requirements.txt

# Start development server
uv run python run_api.py

# Start Celery worker (separate terminal)
uv run celery -A tasks worker --loglevel=info

# Start Redis (required for Celery)
redis-server
```

### Adding New Features

- **API Endpoints**: Add handlers in `api_videos.py` and models in `pydantic_models.py`
- **Background Tasks**: Implement in `tasks.py` with `@celery_app.task` decorator
- **ML Models**: Extend `src/inference.py` and update `model_config_map.json`
- **Analysis**: Add calculations to `src/movement_analysis.py` and visualizations to `src/visualizations.py`

### Environment Variables
```bash
# Required for production
OPENAI_API_KEY=sk-...                    # OpenAI API access
AWS_ACCESS_KEY_ID=AKIA...               # AWS S3 access
AWS_SECRET_ACCESS_KEY=...               # AWS S3 secret
S3_BUCKET=shadow-trainer-pro-keypoints  # Professional keypoints storage

# Optional configuration
CELERY_RESULT_EXPIRES=3600              # Task result expiration (seconds)
MAX_UPLOAD_SIZE=100MB                   # Maximum video upload size
```

### Docker Development
```bash
# Build development container
docker-compose build backend

# Start backend services
docker-compose up backend redis celery

# View logs
docker-compose logs -f backend celery
```

## 🧪 Testing

```bash
# Run API health check
curl http://localhost:8002/health/

# Test video upload
curl -X POST -F "file=@sample.mov" http://localhost:8002/videos/upload-and-process/

# Check processing status
curl http://localhost:8002/videos/{task_id}/status/
```

## 📊 Monitoring

- **Flower Dashboard** - Monitor Celery tasks at `http://localhost:5555`
- **FastAPI Docs** - Interactive API documentation at `http://localhost:8002/docs`
- **Health Endpoints** - System status checks for uptime monitoring
- **Task Logging** - Comprehensive logging for debugging and performance tracking