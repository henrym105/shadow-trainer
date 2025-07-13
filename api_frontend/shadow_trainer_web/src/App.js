import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

// Set default timeout for all axios requests
axios.defaults.timeout = 300000; // 5 minutes

const DEFAULT_S3 = "s3://shadow-trainer-prod/sample_input/henry-mini.mov";
const API_PROCESS_URL = "/api/video/process"; // Back to real processing
const LOCAL_TMP_OUTPUT = "/home/ec2-user/shadow-trainer/api_backend/tmp_api_output";

// const API_URL = "http://www.shadow-trainer.com/api/video/process";


function App() {
  const [videoPath, setVideoPath] = useState(DEFAULT_S3);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResponse(null);
    
    try {
      console.log("Sending request to:", API_PROCESS_URL);
      console.log("With data:", { file: videoPath, model_size: "s" });
      
      const res = await axios.post(API_PROCESS_URL, {
        file: videoPath,
        model_size: "s"
      });
      
      console.log("Response received:", res.data);
      setResponse(res.data);
      
    } catch (err) {
      console.error("Request failed:", err);
      setError(err.response?.data?.error || err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <img src="/assets/Shadow Trainer Logo.png" alt="Shadow Trainer Logo" className="logo" />
        <h3>AI-Powered Motion Analysis</h3>
      </div>

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
                src={`/api/output/${response.output_video_local_path.replace(LOCAL_TMP_OUTPUT + '/', '')}`}
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
