#!/usr/bin/env python3
"""
Test script to verify SageMaker endpoints work correctly
"""
import requests
import json
import time

# Test the API locally first
BASE_URL = "http://localhost:8000"

def test_ping():
    """Test the /ping endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/ping")
        print(f"Ping status: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Ping failed: {e}")
        return False

def test_invocations():
    """Test the /invocations endpoint"""
    try:
        payload = {
            "file": "s3://shadow-trainer-prod/sample_input/pitch_mini2.mp4",
            "model_size": "xs"
        }
        response = requests.post(
            f"{BASE_URL}/invocations",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        print(f"Invocations status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Invocations failed: {e}")
        return False

def test_process_video():
    """Test the original /process_video/ endpoint"""
    try:
        params = {
            "file": "s3://shadow-trainer-prod/sample_input/pitch_mini2.mp4",
            "model_size": "xs"
        }
        response = requests.post(f"{BASE_URL}/process_video/", params=params)
        print(f"Process video status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Process video failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing SageMaker endpoints...")
    
    print("\n1. Testing /ping endpoint:")
    ping_ok = test_ping()
    
    print("\n2. Testing /invocations endpoint:")
    invocations_ok = test_invocations()
    
    print("\n3. Testing original /process_video/ endpoint:")
    process_video_ok = test_process_video()
    
    print(f"\nResults:")
    print(f"  Ping: {'✓' if ping_ok else '✗'}")
    print(f"  Invocations: {'✓' if invocations_ok else '✗'}")
    print(f"  Process Video: {'✓' if process_video_ok else '✗'}")
    
    if ping_ok:
        print("\n✓ API is ready for SageMaker deployment!")
    else:
        print("\n✗ API needs fixes before SageMaker deployment")
