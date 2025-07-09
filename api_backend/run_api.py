import uvicorn

if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8002, reload=True)
