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
├── api_videos.py           # Video processing REST endpoints
├── tasks.py                # Celery background task definitions
├── constants.py            # Configuration constants and settings
├── pydantic_models.py      # API request/response data models
├── requirements.txt        # Python package dependencies
├── run_api.py              # Development server entry point
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
├── checkpoint/             # Pre-trained model weights
│   ├── yolo11x-pose.pt
│   ├── motionagformer-s-h36m.pth.tr
│   └── example_SnellBlake.npy
├── output/                # Processed video results storage
├── uploads/               # Temporary video upload storage
└── sample_videos/         # Test videos and sample outputs
```

## 🔌 API Endpoints

### Video Processing
```bash
POST /videos/upload-and-process/          # Upload video and start processing
GET  /videos/{task_id}/status/            # Check processing status
GET  /videos/{task_id}/download/          # Download processed video
GET  /videos/{task_id}/preview/           # Stream video preview
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
GET  /pro-keypoints/                      # Available professional athletes
GET  /models/                             # Available ML model configurations
GET  /task/{task_id}/status/              # Celery task status
POST /task/{task_id}/cancel/              # Cancel background task
GET  /tasks/active/                       # List all active tasks
```

## 🔄 Processing Pipeline

```
1. Video Upload → FastAPI endpoint validates and stores file
2. Task Queued → Celery worker picks up processing job
3. Video Preprocessing → OpenCV handles rotation, cropping, alignment
4. 2D Pose Detection → YOLO11x extracts joint coordinates per frame
5. 3D Pose Estimation → MotionAGFormer lifts 2D points to 3D space
6. Motion Alignment → Align user and pro athlete hip orientations
7. Biomechanical Analysis → Calculate joint angles, velocities, timing
8. Visualization → Generate 3D animations and analysis charts
9. AI Feedback → OpenAI API creates personalized coaching insights
10. Results Storage → Save outputs to filesystem with metadata
```

## 🧠 Key Components

### FastAPI Service (`api_service.py`)
- CORS-enabled REST API with automatic OpenAPI documentation
- File upload handling with size and format validation
- Health check endpoints for system monitoring
- Integration with Celery distributed task queue

### Celery Workers (`tasks.py`)
- Background video processing pipeline execution
- Redis-backed task queue with retry logic
- Progress tracking and real-time status updates
- Error handling and graceful failure recovery

### ML Inference Pipeline (`src/inference.py`)
- YOLO11x 2D pose detection with confidence filtering
- MotionAGFormer 3D pose estimation from 2D keypoints
- Video preprocessing including rotation and cropping
- Frame-by-frame keypoint extraction and temporal smoothing

### Motion Analysis (`src/movement_analysis.py`)
- Baseball-specific biomechanical calculations
- Joint angle analysis (hip, shoulder, elbow, wrist mechanics)
- Timing analysis compared to professional athlete data
- Hip-shoulder separation and rotational velocity calculations
- OpenAI integration for generating personalized coaching feedback

### Visualization Engine (`src/visualizations.py`)
- 3D skeleton animation generation using matplotlib
- Interactive analysis chart creation (joint angles, velocities)
- Comparative visualization between user and professional motions
- Export capabilities for PNG charts and analysis plots

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

**New API Endpoints:**
- Add route handlers in `api_videos.py`
- Define request/response models in `pydantic_models.py`
- Update OpenAPI documentation automatically

**Background Processing Tasks:**
- Implement new tasks in `tasks.py` with `@celery_app.task` decorator
- Add error handling and progress reporting
- Update task status tracking in frontend

**ML Model Integration:**
- Extend `src/inference.py` with new model loading and inference
- Add model configurations to `model_config_map.json`
- Update checkpoint management in `checkpoint/` directory

**Biomechanical Analysis:**
- Add new calculations to `src/movement_analysis.py`
- Create visualization functions in `src/visualizations.py`
- Update OpenAI prompts for enhanced feedback

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