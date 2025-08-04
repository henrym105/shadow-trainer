# Shadow Trainer - AI-Powered Motion Analysis

**Professional athletic motion analysis platform** that uses advanced computer vision and AI to analyze baseball pitching mechanics. Upload videos of your pitching motion and get detailed biomechanical analysis compared to professional athletes.

## 🏗️ System Architecture

Shadow Trainer is deployed as a **monolithic application on EC2** using Docker Compose to orchestrate multiple services:

### Monolithic Deployment Stack
- **EC2 Instance** - Single server hosting all services via Docker Compose
- **FastAPI Backend** - Python API server handling requests and ML processing
- **React Frontend** - JavaScript SPA for user interface
- **Celery + Redis** - Distributed task queue for background video processing
- **Nginx** - Reverse proxy and static file serving
- **Docker Compose** - Multi-container orchestration on single EC2 instance

### Core Technology Stack
**Backend (Python)**
- **FastAPI** - High-performance API server with automatic OpenAPI docs
- **Celery + Redis** - Background task processing for video analysis
- **YOLO11x-pose** - 2D human pose estimation
- **MotionAGFormer** - 3D pose estimation from 2D keypoints
- **OpenAI API** - AI-powered biomechanical coaching feedback

**Frontend (JavaScript)**
- **React** - Component-based UI framework
- **Three.js** - 3D visualization and skeleton rendering
- **CSS Modules** - Scoped styling for components

## 🚀 Quick Start

### Prerequisites
- **Linux EC2 instance** (g4dn.xlarge recommended for nvidia Cuda gpu acceleration)
- **Python 3.9+** with [uv](https://github.com/astral-sh/uv) package manager
- **Node.js 18+** and npm
- **Docker & Docker Compose**
- **OpenAI API key** (optional, for AI coaching feedback)

### Installation & Setup
```bash
# Clone repository
git clone https://github.com/henrym105/shadow-trainer.git
cd shadow-trainer

# Install Python dependencies with UV
./scripts/install_setup_uv.sh

# Install frontend dependencies
cd api_frontend && npm install && cd ..

# Configure environment (optional)
echo "OPENAI_API_KEY=your_key_here" > .env.prod

# Start production services
make prod
```

### Command Reference
```bash
make dev          # Start development environment
make prod         # Start production environment  
make prod-build   # Build and start with 3 workers
make stop         # Stop all services
make health       # System health check
make clean        # Remove containers and volumes
make dev-logs     # View development logs
make prod-logs    # View production logs
```

## 🎯 Platform Features

### Video Processing Workflow
1. **Video Upload** - Drag & drop interface with format validation
2. **2D Pose Detection** - YOLO11x extracts 2D joint coordinates 
3. **3D Pose Estimation** - MotionAGFormer converts to 3D keypoints
4. **Motion Alignment** - Aligns user and pro motions at hip orientation
5. **Biomechanical Analysis** - Calculates joint angles, velocities, timing
6. **AI Coaching** - OpenAI generates personalized feedback

### Advanced Analytics
- **Joint-by-joint Analysis** - Hip, shoulder, elbow, wrist mechanics
- **Timing Analysis** - Phase comparisons with professional athletes
- **Angular Velocity Tracking** - Throwing arm speed analysis  
- **Hip-Shoulder Separation** - Core rotation mechanics
- **Interactive Visualizations** - Matplotlib-generated analysis charts

### 3D Visualization Engine
- **Real-time Skeleton Rendering** - WebGL-based 3D viewer
- **Dual Skeleton Display** - User vs professional comparison
- **Playback Controls** - Frame-by-frame analysis capabilities
- **Multiple Camera Angles** - Turntable rotation for different perspectives

## 🔌 Backend API Reference

### Core Video Processing
```bash
POST /videos/upload-and-process/          # Upload video, start processing
GET  /videos/{task_id}/status/            # Check processing status
GET  /videos/{task_id}/download/          # Download processed video
GET  /videos/{task_id}/preview/           # Stream video preview
POST /videos/{task_id}/terminate/         # Cancel processing job
```

### 3D Data & Visualization  
```bash
GET  /videos/{task_id}/keypoints/user/    # User 3D keypoints (JSON)
GET  /videos/{task_id}/keypoints/pro/     # Pro athlete keypoints (JSON)  
GET  /videos/{task_id}/analysis-plots/    # Movement analysis charts
POST /videos/{task_id}/generate-evaluation/ # AI coaching feedback
```

### System & Configuration
```bash
GET  /health/                             # System health check
GET  /pro-keypoints/                      # Available pro athletes
GET  /models/                             # Available ML models
```

### Task Management (Celery)
```bash
GET  /task/{task_id}/status/              # Celery task status
POST /task/{task_id}/cancel/              # Cancel background task
GET  /tasks/active/                       # List active tasks
```

## 🏗️ Technical Architecture

### Backend Infrastructure
- **FastAPI 0.116+** - Modern Python web framework with async support
- **Celery 5.5+** - Distributed task queue with Redis broker
- **Redis 6.0+** - In-memory data store for task queuing and caching
- **Nginx** - Reverse proxy and static file serving
- **Docker Compose** - Multi-container orchestration
- **AWS S3** - Professional athlete keypoint dataset storage

### Machine Learning Stack
- **PyTorch 2.3+** - Deep learning framework
- **Ultralytics YOLO11x** - State-of-the-art pose estimation
- **MotionAGFormer** - 3D pose lifting from 2D keypoints  
- **OpenCV 4.11+** - Computer vision and video processing
- **NumPy/SciPy** - Numerical computing for motion analysis
- **Matplotlib** - Statistical visualization and chart generation

### Backend Processing Flow
```
1. Video Upload → FastAPI endpoint
2. Task Queued → Redis + Celery  
3. Video Preprocessing → OpenCV (rotation, cropping)
4. 2D Pose Detection → YOLO11x model inference
5. 3D Pose Estimation → MotionAGFormer neural network
6. Motion Alignment → Hip-facing alignment algorithm
7. Biomechanical Analysis → Joint angles, velocities, timing
8. Visualization Generation → 3D animations + analysis plots
9. AI Feedback → OpenAI API integration
10. Results Storage → Local filesystem + metadata
```

### Container Architecture
```bash
# Production Stack (docker-compose.prod.yml)
├── nginx:latest              # Reverse proxy (port 80/443)  
├── react-app:custom          # Frontend build (port 3000)
├── fastapi-backend:custom    # API server (port 8002)
├── redis:7-alpine           # Task broker (port 6379)
├── celery-worker:custom     # Processing workers (×3)
└── flower:latest            # Task monitoring (port 5555)
```

## 📁 Project Structure

```
shadow-trainer/                 # Monolithic project root
├── api_backend/               # Python FastAPI backend (see api_backend/README.md)
├── api_frontend/              # React JavaScript frontend (see api_frontend/README.md)
├── docker-compose.yml         # Development services
├── docker-compose.prod.yml    # Production services
└── Makefile                  # Build and deployment commands
```

### Environment Configuration
```bash
# .env.prod (production)
OPENAI_API_KEY=sk-...
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET=shadow-trainer-pro-keypoints
CELERY_RESULT_EXPIRES=3600
```