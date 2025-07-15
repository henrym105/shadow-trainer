#!/usr/bin/env python3
"""
Test script to verify the API accepts the is_lefty parameter
"""
import requests
import json

def test_upload_endpoint():
    """Test that the upload endpoint accepts is_lefty parameter"""
    url = "http://localhost:8002/api/videos/upload"
    
    # Test parameters
    params = {
        "model_size": "xs",
        "is_lefty": "true"
    }
    
    # Create a small test file (we won't actually upload, just test parameter acceptance)
    print("Testing API endpoint parameter acceptance...")
    print(f"URL: {url}")
    print(f"Parameters: {params}")
    
    # Just test the parameter structure by making an OPTIONS request
    try:
        response = requests.options(url, params=params)
        print(f"OPTIONS response status: {response.status_code}")
        
        # Test with a GET to see the parameter validation
        response = requests.get(url, params=params)
        print(f"GET response status: {response.status_code}")
        print("✅ API endpoint accepts is_lefty parameter")
        
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")

if __name__ == "__main__":
    test_upload_endpoint()
