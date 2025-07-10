<!-- # Shadow Trainer -->

<p align="center">
    <h1 align="center"><strong>Shadow Trainer</strong></h1>
<p align="center">
  <img src="api_frontend/assets/Shadow Trainer Logo.png" alt="Shadow Trainer Logo" width="300"/>

Shadow Trainer is an end-to-end video processing and 3D human pose estimation platform. It provides a FastAPI backend for inference, a Streamlit frontend for visualization, and scripts for cloud and local deployment. The system supports local files and S3 video sources, and outputs pose data and processed videos.

<h1 align="center">
    <a href="http://www.shadow-trainer.com">www.shadow-trainer.com</a></h1>


## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [API Usage](#api-usage)
- [Frontend Usage](#frontend-usage)
- [Scripts](#scripts)
- [Configuration](#configuration)
- [Development](#development)
- [License](#license)

---


## Features
- 2D/3D human pose estimation from video (local or S3)
- FastAPI backend for inference and S3 integration
- Streamlit frontend for video upload, visualization, and API interaction
- Docker support for reproducible deployment
- Model configuration for multiple sizes (xs, s, b, l)
- Output includes processed video, pose2D/3D, and raw keypoints


<p align="center">
    <img src="api_frontend/assets/example_output.gif" alt="Sample Output GIF" width="500"/>
    <br/>
    <em>Example: Model inference and output visualization</em>
</p>


## Project Structure
```
├── api_backend/         # FastAPI backend and inference logic
│   ├── api_service.py   # Main FastAPI app
│   ├── run_api.py       # Entrypoint for backend
│   └── ...
├── api_frontend/        # Streamlit frontend apps
│   ├── streamlit_app.py # Main Streamlit app
│   └── ...
├── src/                 # Core model, preprocessing, and utils
├── scripts/             # Shell scripts for setup, curl, and deployment
├── shared/              # Shared config
├── Dockerfile           # Docker build file
├── pyproject.toml       # Python dependencies
├── uv.lock              # uv dependency lockfile
└── README.md            # This file
```

## Architecture Notes

### Monolith Architecture (Current)

Shadow Trainer is currently implemented as a monolithic application, meaning the backend API, frontend, and core processing logic are all contained within a single codebase and deployed together. This approach is intentional and well-suited for the early stages of the project for several reasons:

- **Simplicity:** A monolith is easier to develop, test, and deploy when the team is small and requirements are evolving rapidly.
- **Faster Iteration:** Changes to the API, frontend, and models can be made and tested together without the overhead of managing multiple repositories or deployment pipelines.
- **Lower Operational Overhead:** There is no need to manage inter-service communication, service discovery, or distributed tracing, which reduces complexity and cost.
- **Unified Codebase:** All logic, configuration, and dependencies are in one place, making onboarding and debugging more straightforward.

### Future Scalability: Microservices & Kubernetes

As the project matures and usage grows, it may make sense to migrate to a microservices architecture. This would involve splitting the backend API, model inference, frontend, and possibly other components (e.g., video processing, S3 integration) into separate services. Benefits of this approach include:

- **Independent Scaling:** Each service can be scaled based on its own resource needs (e.g., GPU inference vs. web frontend).
- **Technology Flexibility:** Different services can use the most appropriate language, framework, or runtime for their task.
- **Fault Isolation:** Failures in one service are less likely to impact the entire system.
- **Easier CI/CD:** Teams can deploy and update services independently.

If/when the project outgrows the monolith, container orchestration platforms like Kubernetes (K8s) can be used to manage deployment, scaling, and networking of microservices. This transition is a common path for production systems as they reach larger scale and require higher reliability.

For now, the monolithic approach provides the best balance of speed, simplicity, and maintainability for rapid development and experimentation.

## Quickstart

### 1. Install Dependencies

#### Using [uv](https://github.com/astral-sh/uv) (Recommended)
```bash
chmod +x ./scripts/install_setup_uv.sh
./scripts/install_setup_uv.sh
```
This will install `uv` if needed, sync dependencies, and activate the virtual environment.



### 2. Start All Services (Recommended)

To start both the backend API and frontend Streamlit app in separate tmux panes (recommended for production or cloud deployment):

```bash
chmod +x ./start_all.sh
./start_all.sh
```
This will:
- Start the FastAPI backend on port **8002** (serving at `http://www.shadow-trainer.com/api/`)
- Start the Streamlit frontend on port **8000** (serving at `http://www.shadow-trainer.com/`)
- Open both in a tmux session for easy monitoring

**Note:** Ensure ports 8000 and 8002 are open in your AWS security group and mapped to your domain.

#### Manual Start (Alternative)

**Backend:**
```bash
cd api_backend
uv run python run_api.py
# or use Docker:
# docker build -t shadow-trainer .
# docker run -p 8002:8002 shadow-trainer
```
**Frontend:**
```bash
cd api_frontend
uv run streamlit run streamlit_app.py --server.port 8000 --server.enableCORS false
```
The frontend will be available at [http://www.shadow-trainer.com/](http://www.shadow-trainer.com/) and the API at [http://www.shadow-trainer.com/api/](http://www.shadow-trainer.com/api/)

---

## API Usage

### Process a Video (Local or S3)

**Endpoint:** `POST /process_video/`

**Parameters:**
- `file`: Path to local video or S3 URI (e.g. `s3://bucket/video.mp4`)
- `model_size`: Model size (`xs`, `s`, `b`, `l`)

**Example (S3):**
```bash
curl -X POST "http://localhost:8002/process_video/" \
    -H "accept: application/json" \
    --get \
    --data-urlencode "file=s3://shadow-trainer-prod/sample_input/henry1_full.mov" \
    --data-urlencode "model_size=s"
```

**Example (Local File):**
```bash
curl -X POST "http://localhost:8002/process_video/" \
    -H "accept: application/json" \
    -F "file=@/path/to/video.mp4" \
    -F "model_size=xs"
```

See `scripts/example_curl_*.sh` for more examples.

---

## Frontend Usage

The Streamlit app allows you to:
- Upload a video or enter an S3 path
- Select model size
- View processed video and pose outputs

Run with:
```bash
cd api_frontend
uv run streamlit run streamlit_app.py
```

---

## Scripts

- `scripts/install_setup_uv.sh` – Install uv and sync dependencies
- `scripts/setup_ec2_linux.sh` – Setup for AWS EC2 (Amazon Linux)
- `scripts/create_nginx_config.sh` – NGINX install and config helper
- `scripts/example_curl_*.sh` – Example API usage with curl
- `scripts/test_url.sh` – Test NGINX and site availability

---

## Configuration

- Model configs: `src/configs/`, `src/hrnet/experiments/`
- Checkpoints: `src/checkpoint/`, `api_backend/checkpoint/`
- Output: `tmp_api_output/`, `tmp_save_for_later/`

---

## Development

- Backend: FastAPI (`api_backend/`)
- Frontend: Streamlit (`api_frontend/`)
- Core models/utils: `api_backend/src/`
- Shared config: `shared/`

### Tips
- Use the provided scripts for setup and API testing
- For cloud/S3, ensure AWS credentials are configured
- For Docker, see `Dockerfile` and scripts

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.