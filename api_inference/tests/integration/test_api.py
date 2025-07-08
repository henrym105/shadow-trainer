"""
Integration tests for Shadow Trainer API endpoints.
"""
import pytest
from fastapi.testclient import TestClient

def test_root_endpoint(client: TestClient):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data

def test_health_endpoint(client: TestClient):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_api_health_endpoint(client: TestClient):
    """Test the API health check endpoint."""
    response = client.get("/api/v1/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_frontend_ui_endpoint(client: TestClient):
    """Test the frontend UI endpoint."""
    response = client.get("/frontend/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_api_docs_accessible(client: TestClient):
    """Test that API documentation is accessible."""
    response = client.get("/api/v1/docs")
    assert response.status_code == 200

def test_sample_videos_endpoint(client: TestClient):
    """Test the sample videos endpoint."""
    response = client.get("/frontend/sample_videos")
    assert response.status_code == 200
    data = response.json()
    assert "videos" in data
    assert isinstance(data["videos"], list)
