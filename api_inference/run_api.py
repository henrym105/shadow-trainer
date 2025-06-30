import uvicorn

if __name__ == "__main__":
    # uvicorn.run("api_inference.api_service:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)
