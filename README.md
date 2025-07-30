# Shadow Trainer - AI-Powered Motion Analysis

Video processing platform for athletic motion analysis using YOLO pose estimation and AI feedback. Compare your movement to professional athletes with 3D visualization.

## Quick Start

### Prerequisites
- Linux EC2 instance 
- Python 3.9+ with [uv](https://github.com/astral-sh/uv)
- Node.js 18+ and npm
- OpenAI API key (optional, for AI feedback)

### Setup
```bash
# Clone and install
git clone https://github.com/henrym105/shadow-trainer.git
cd shadow-trainer
./scripts/install_setup_uv.sh
cd api_frontend && npm install

# Configure OpenAI (optional)
echo "OPENAI_API_KEY=your_key_here" > .env

# Start services
make dev        # Development
make prod       # Production
```

### Commands
- `make dev` / `make prod` — Start services
- `make stop` — Stop all services  
- `make health` — Health check
- `make dev-logs` / `make prod-logs` — View logs

## Features

### Core Functionality
- **Video Upload** — Drag & drop with validation
- **Motion Analysis** — YOLO 2D + MotionAGFormer 3D pose estimation
- **Professional Comparison** — Compare against pro athlete movements
- **Real-time Processing** — Background jobs with progress updates

### 3D Visualization
- Interactive skeleton viewer at `/visualize/{task_id}`
- Playback controls (play/pause, speed, frame scrubbing)
- Toggle user/pro skeletons independently
- Turntable rotation for different viewing angles

### AI Feedback
- Generate detailed movement analysis using OpenAI
- Joint-by-joint biomechanical evaluation
- Coaching insights and improvement suggestions

## API Endpoints

### Core
- `POST /api/videos/upload` — Upload and process video
- `GET /api/videos/{id}/status` — Processing status
- `GET /api/videos/{id}/download` — Download result
- `GET /api/videos/{id}/preview` — Stream preview

### Visualization
- `GET /api/videos/{id}/keypoints/user` — User 3D keypoints
- `GET /api/videos/{id}/keypoints/pro` — Pro athlete keypoints
- `POST /api/videos/{id}/generate-evaluation` — Generate AI feedback

## Architecture

**Stack:** React frontend + FastAPI backend + Nginx proxy + Redis/Celery
**Deployment:** Docker Compose on EC2

```mermaid
graph TB
    subgraph "Client Layer"
        USER[User Browser]
    end
    
    subgraph "Frontend Layer"
        REACT[React App<br/>Port 3000]
        NGINX[Nginx Proxy<br/>Port 80/443]
    end
    
    subgraph "Backend Services"
        API[FastAPI Service<br/>Port 8002]
        REDIS[Redis<br/>Port 6379]
        CELERY[Celery Workers]
    end
    
    subgraph "ML Pipeline"
        YOLO[YOLO 2D Pose<br/>Detection]
        MOTION[MotionAGFormer<br/>3D Pose Estimation]
        ANALYSIS[Joint Analysis<br/>& Comparison]
    end
    
    subgraph "Storage"
        S3[AWS S3<br/>Pro Keypoints]
        OUTPUTS[Local Storage<br/>Processed Videos<br/>& Keypoints]
    end
    
    subgraph "External Services"
        OPENAI[OpenAI API<br/>Movement Analysis]
    end
    
    %% User interactions
    USER -->|HTTP Requests| NGINX
    NGINX -->|Serve Static Files| REACT
    NGINX -->|API Proxy /api/*| API
    
    %% Frontend to Backend API calls
    REACT -.->|Video Upload<br/>Status Checks<br/>Download Results| API
    
    %% Backend processing flow
    API -->|Queue Tasks| REDIS
    REDIS -->|Task Distribution| CELERY
    CELERY -->|Process Videos| YOLO
    YOLO -->|2D Keypoints| MOTION
    MOTION -->|3D Keypoints| ANALYSIS
    
    %% Data storage and retrieval
    API -->|Load Pro Data| S3
    CELERY -->|Save Results| OUTPUTS
    API -->|Serve Files| OUTPUTS
    
    %% AI feedback generation
    ANALYSIS -.->|Generate Feedback| OPENAI
    
    %% Key endpoints
    API -.->|POST /videos/upload<br/>GET /videos/ID/status<br/>GET /videos/ID/download<br/>GET /videos/ID/keypoints/*| REACT
    
    classDef frontend fill:#e1f5fe,color:#000000
    classDef backend fill:#f3e5f5,color:#000000
    classDef ml fill:#e8f5e8,color:#000000
    classDef storage fill:#fff3e0,color:#000000
    classDef external fill:#fce4ec,color:#000000
    
    class USER,REACT,NGINX frontend
    class API,REDIS,CELERY backend
    class YOLO,MOTION,ANALYSIS ml
    class S3,OUTPUTS storage
    class OPENAI external
```


## Development

### Project Structure
```
shadow-trainer/
├── api_backend/           # FastAPI + ML pipeline
├── api_frontend/          # React app
└── Makefile               # Commands
```

### Adding Features
- **Backend:** Modify `api_service.py`, `tasks.py`
- **Frontend:** Update components in `src/components/`
- **Infrastructure:** Update Makefile/nginx configs