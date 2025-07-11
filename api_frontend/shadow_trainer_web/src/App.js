import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

const DEFAULT_S3 = "s3://shadow-trainer-prod/sample_input/henry-mini.mov";
const API_URL = "http://localhost:8002/video/process";

function App() {
  const [videoPath, setVideoPath] = useState(DEFAULT_S3);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [inputVideoAvailable, setInputVideoAvailable] = useState(false);
  const [inputVideoUrl, setInputVideoUrl] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResponse(null);
    setInputVideoAvailable(false);
    setInputVideoUrl("");
    try {
      // For S3, predict the local URL
      if (videoPath.startsWith("s3://")) {
        const basename = videoPath.split("/").pop();
        const url = `http://localhost:8002/output/${basename}`;
        setInputVideoUrl(url);
      } else if (videoPath.startsWith("http")) {
        setInputVideoUrl(videoPath);
        setInputVideoAvailable(true);
      }
      const res = await axios.post(
        API_URL,
        { file: videoPath, model_size: "xs" },
        { headers: { "accept": "application/json", "Content-Type": "application/json" }, timeout: 1000 * 60 * 5 }
      );
      setResponse(res.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  // Poll for S3 input video availability
  useEffect(() => {
    let intervalId;
    if (loading && videoPath.startsWith("s3://") && inputVideoUrl) {
      intervalId = setInterval(async () => {
        try {
          const res = await fetch(inputVideoUrl, { method: "HEAD" });
          if (res.ok) {
            setInputVideoAvailable(true);
            clearInterval(intervalId);
          }
        } catch (e) {
          // Not available yet
        }
      }, 1000);
    }
    return () => intervalId && clearInterval(intervalId);
  }, [loading, videoPath, inputVideoUrl]);

  return (
    <div className="container">
      <div className="header">
        <img src="/assets/Shadow Trainer Logo.png" alt="Shadow Trainer Logo" className="logo" />
        {/* <h1>Shadow Trainer</h1> */}
        <h3>AI-Powered Motion Analysis</h3>
      </div>
      {/* Show input video preview while processing */}
      {loading && (
        <div className="input-preview">
          <h5>Input Video Preview:</h5>
          {inputVideoUrl && inputVideoAvailable ? (
            <video controls src={inputVideoUrl} style={{ maxWidth: "100%" }} />
          ) : (
            <div style={{ color: '#888', fontStyle: 'italic' }}>
              {videoPath.startsWith('s3://')
                ? "Waiting for input video to be available..."
                : "Loading preview..."}
            </div>
          )}
        </div>
      )}
      <form className="main-form" onSubmit={handleSubmit}>
        <label htmlFor="videoPath">Video S3 Path</label>
        <input
          id="videoPath"
          type="text"
          value={videoPath}
          onChange={(e) => setVideoPath(e.target.value)}
          placeholder="Paste your S3 video path here"
        />
        <button type="submit" disabled={loading} className="cta-btn">
          {loading ? "Processing..." : "Create Your Shadow"}
        </button>
      </form>
      {error && <div className="error">{error}</div>}
      {response && (
        <div className="response">
          <h4>API Response</h4>
          <pre>{JSON.stringify(response, null, 2)}</pre>
          {response.output_video_local_path ? (
            <div>
              <h5>Output Video Preview:</h5>
              <video
                controls
                src={`http://localhost:8002/output/${response.output_video_local_path.split('/tmp_api_output/')[1]}`}
                style={{ maxWidth: "100%" }}
              />
            </div>
          ) : response.output_video_s3_url ? (
            <div>
              <h5>Output Video S3 URL:</h5>
              <a href={response.output_video_s3_url} target="_blank" rel="noopener noreferrer">
                {response.output_video_s3_url}
              </a>
            </div>
          ) : null}
        </div>
      )}
      <footer>
        &copy; 2025 Shadow Trainer. All rights reserved.
      </footer>
    </div>
  );
}

export default App;
