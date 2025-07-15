/**
 * Shadow Trainer - Main Application Component
 * AI-Powered Motion Analysis for Athletic Performance
 */

import React, { useState, useCallback } from "react";
import { VideoAPI, useJobPolling, APIError } from "./services/videoApi";
import FileUpload from "./components/FileUpload";
import ProgressBar from "./components/ProgressBar";
import VideoResult from "./components/VideoResult";
import ProKeypointsSelector from "./components/ProKeypointsSelector";
import ThreeJSkeleton from "./components/ThreeJSkeleton";
import "./App.css";

function App() {
  // State management
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [modelSize, setModelSize] = useState('xs');
  const [isLefty, setIsLefty] = useState(false); // New state for handedness preference
  const [selectedProFile, setSelectedProFile] = useState(""); // New state for ProKeypoints file
  const [resultView, setResultView] = useState('both'); // 'threejs', 'video', 'both'

  // Handle file selection from FileUpload component
  const handleFileSelect = useCallback((file, error) => {
    setSelectedFile(file);
    setUploadError(error);
    
    // Clear previous job state when new file selected
    if (file) {
      setJobId(null);
      setJobStatus(null);
    }
  }, []);

  // Handle job status updates from polling
  const handleStatusUpdate = useCallback((status) => {
    setJobStatus(status);
    console.log('Job status update:', status);
  }, []);

  // Use polling hook to automatically check job status
  const { isPolling, error: pollingError } = useJobPolling(jobId, handleStatusUpdate);

  // Handle video upload and processing
  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadError('Please select a video file first');
      return;
    }

    // Validate file before upload
    const validation = VideoAPI.validateFile(selectedFile);
    if (!validation.isValid) {
      setUploadError(validation.error);
      return;
    }

    setIsUploading(true);
    setUploadError(null);

    try {
      console.log('Starting upload...', selectedFile.name);
      const response = await VideoAPI.uploadVideo(selectedFile, modelSize, isLefty, selectedProFile, resultView); // Pass resultView
      
      console.log('Upload successful:', response);
      setJobId(response.job_id);
      
      // Initial status
      setJobStatus({
        job_id: response.job_id,
        status: 'queued',
        progress: 0,
        message: 'Upload successful, processing queued...'
      });

    } catch (error) {
      console.error('Upload failed:', error);
      let errorMessage = 'Upload failed. Please try again.';
      
      if (error instanceof APIError) {
        errorMessage = error.detail || error.message;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      setUploadError(errorMessage);
    } finally {
      setIsUploading(false);
    }
  };

  // Reset to initial state
  const handleReset = () => {
    setSelectedFile(null);
    setUploadError(null);
    setJobId(null);
    setJobStatus(null);
    setIsUploading(false);
    setIsLefty(false); // Reset handedness preference
    setSelectedProFile(""); // Reset ProKeypoints file
  };

  // Determine current application state
  const getAppState = () => {
    if (jobStatus?.status === 'completed') return 'completed';
    if (jobStatus?.status === 'failed') return 'failed';
    if (jobId && (jobStatus?.status === 'processing' || jobStatus?.status === 'queued')) return 'processing';
    return 'upload';
  };

  const appState = getAppState();

  return (
    <div className="app">
      <div className="app-container">
        {/* Header */}
        <header className="app-header">
          <div className="logo-section">
            <img src="/assets/Shadow Trainer Logo.png" alt="Shadow Trainer" className="logo" />
            <div className="logo-text">
              <h1>Shadow Trainer</h1>
              <p>AI-Powered Motion Analysis</p>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="app-main">
          {appState === 'upload' && (
            <div className="upload-section">
              <div className="section-header">
                <h2>Upload Your Training Video</h2>
                <p>Get detailed motion analysis and pose estimation for your athletic performance</p>
              </div>

              {/* New: Result view selection */}
              <div className="result-view-selection" style={{ marginBottom: '1rem' }}>
                <label style={{ marginRight: '1rem' }}><strong>Result View:</strong></label>
                <label style={{ marginRight: '1rem' }}>
                  <input type="radio" name="resultView" value="threejs" checked={resultView === 'threejs'} onChange={() => setResultView('threejs')} />
                  3D Skeleton Only
                </label>
                <label style={{ marginRight: '1rem' }}>
                  <input type="radio" name="resultView" value="video" checked={resultView === 'video'} onChange={() => setResultView('video')} />
                  2D/3D Video Only
                </label>
                <label>
                  <input type="radio" name="resultView" value="both" checked={resultView === 'both'} onChange={() => setResultView('both')} />
                  Both
                </label>
              </div>

              <FileUpload
                onFileSelect={handleFileSelect}
                disabled={isUploading}
              />

              {(uploadError || pollingError) && (
                <div className="error-message">
                  <span className="error-icon">⚠️</span>
                  {uploadError || pollingError}
                </div>
              )}

              {selectedFile && (
                <div className="upload-controls">
                  <div className="model-selection">
                    <label htmlFor="model-size">Analysis Quality:</label>
                    <select
                      id="model-size"
                      value={modelSize}
                      onChange={(e) => setModelSize(e.target.value)}
                      disabled={isUploading}
                    >
                      <option value="xs">Fast (XS) - 30-60 seconds</option>
                      <option value="s">Balanced (S) - 60-90 seconds</option>
                      <option value="m">High Quality (M) - 90-120 seconds</option>
                    </select>
                  </div>

                  <div className="handedness-selection">
                    <label htmlFor="handedness-toggle">Dominant Hand:</label>
                    <div className="toggle-container">
                      <span className={`toggle-label ${!isLefty ? 'active' : ''}`}>Right</span>
                      <label className="toggle-switch">
                        <input
                          type="checkbox"
                          checked={isLefty}
                          onChange={(e) => setIsLefty(e.target.checked)}
                          disabled={isUploading}
                        />
                        <span className="slider"></span>
                      </label>
                      <span className={`toggle-label ${isLefty ? 'active' : ''}`}>Left</span>
                    </div>
                  </div>

                  <ProKeypointsSelector
                    onSelect={setSelectedProFile}
                    disabled={isUploading}
                  />

                  <button
                    className="upload-btn"
                    onClick={handleUpload}
                    disabled={isUploading || !selectedFile}
                  >
                    {isUploading ? (
                      <>
                        <span className="btn-spinner"></span>
                        Uploading...
                      </>
                    ) : (
                      <>
                        <span className="btn-icon">🚀</span>
                        Start Analysis
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          )}

          {appState === 'processing' && jobStatus && (
            <div className="processing-section">
              <div className="section-header">
                <h2>Processing Your Video</h2>
                <p>Our AI is analyzing your movement patterns and generating pose estimations</p>
              </div>

              <ProgressBar
                progress={jobStatus.progress}
                status={jobStatus.status}
                message={jobStatus.message}
                animated={true}
              />

              <div className="processing-info">
                <div className="processing-steps">
                  <div className={`step ${jobStatus.progress >= 20 ? 'completed' : 'pending'}`}>
                    <span className="step-icon">👁️</span>
                    <span className="step-text">2D Keypoint Detection</span>
                  </div>
                  <div className={`step ${jobStatus.progress >= 50 ? 'completed' : 'pending'}`}>
                    <span className="step-icon">🎯</span>
                    <span className="step-text">3D Pose Estimation</span>
                  </div>
                  <div className={`step ${jobStatus.progress >= 80 ? 'completed' : 'pending'}`}>
                    <span className="step-icon">🎨</span>
                    <span className="step-text">Visualization Generation</span>
                  </div>
                  <div className={`step ${jobStatus.progress >= 100 ? 'completed' : 'pending'}`}>
                    <span className="step-icon">🎬</span>
                    <span className="step-text">Video Compilation</span>
                  </div>
                </div>

                <p className="job-id">Job ID: {jobStatus.job_id}</p>
              </div>
            </div>
          )}

          {appState === 'completed' && jobStatus && (
            <VideoResult
              jobId={jobStatus.job_id}
              originalFilename={selectedFile?.name || 'video'}
              previewUrl={VideoAPI.getPreviewUrl(jobStatus.job_id)}
              downloadUrl={VideoAPI.getDownloadUrl(jobStatus.job_id)}
              resultView={resultView}
            />
          )}

          {appState === 'failed' && jobStatus && (
            <div className="error-section">
              <div className="error-content">
                <div className="error-icon">😞</div>
                <h3>Processing Failed</h3>
                <p>We encountered an issue while processing your video.</p>
                {jobStatus.error && (
                  <div className="error-details">
                    <strong>Error:</strong> {jobStatus.error}
                  </div>
                )}
                <button className="retry-btn" onClick={handleReset}>
                  <span className="btn-icon">🔄</span>
                  Try Again
                </button>
              </div>
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="app-footer">
          <div className="footer-content">
            <p>&copy; 2025 Shadow Trainer. All rights reserved.</p>
            <div className="footer-links">
              <a href="#privacy">Privacy Policy</a>
              <a href="#terms">Terms of Service</a>
              <a href="#support">Support</a>
            </div>
          </div>
        </footer>

        {/* Reset Button - Always available */}
        {(jobId || selectedFile) && (
          <button className="reset-btn floating" onClick={handleReset}>
            <span className="btn-icon">🏠</span>
            Start Over
          </button>
        )}
      </div>
    </div>
  );
}

export default App;
