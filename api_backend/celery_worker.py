"""
Worker startup script that properly registers tasks.
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from celery_app import celery_app

# Register tasks for the worker
def register_tasks():
    """Register tasks when running as a worker."""
    try:
        # Import and register tasks
        from tasks import video_processing
        print("✅ Tasks registered successfully")
    except ImportError as e:
        print(f"❌ Failed to import tasks: {e}")
        raise

if __name__ == '__main__':
    register_tasks()
    celery_app.start()
