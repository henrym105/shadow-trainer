"""
Test configuration and fixtures for Shadow Trainer tests.
"""
import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from shadow_trainer.main import create_app
from shadow_trainer.config import Settings

@pytest.fixture
def test_app():
    """Create a test FastAPI application."""
    # Override settings for testing
    test_settings = Settings(
        environment="testing",
        debug=True,
        database_url="sqlite:///:memory:",
        upload_dir=tempfile.mkdtemp(),
        output_dir=tempfile.mkdtemp(),
    )
    
    app = create_app()
    return app

@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)

@pytest.fixture
def sample_video_file():
    """Create a sample video file for testing."""
    video_content = b"fake_video_content"
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_content)
        return Path(f.name)

@pytest.fixture
def cleanup_temp_files():
    """Cleanup temporary files after tests."""
    temp_files = []
    yield temp_files
    
    # Cleanup
    for file_path in temp_files:
        if Path(file_path).exists():
            Path(file_path).unlink()
