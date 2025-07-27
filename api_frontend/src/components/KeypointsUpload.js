import React, { useState } from 'react';
import FileUpload from './FileUpload';
import ProgressBar from './ProgressBar';
import VideoAPI, { useJobPolling } from '../services/videoApi';
import './KeypointsUpload.css';

const KeypointsUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [modelSize, setModelSize] = useState('xs');

  // Poll job status
  useJobPolling(taskId, setJobStatus, 2000);

  // Handle file selection
  const handleFileSelect = (file, error) => {
    setSelectedFile(file);
    setUploadError(error);
  };

  // Handle upload for 3D keypoints extraction
  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setUploadError(null);
    try {
      const res = await VideoAPI.upload3DKeypoints(selectedFile, modelSize);
      setTaskId(res.task_id);
      setJobStatus({ status: 'queued', progress: 0 });
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  // Reset for new upload
  const handleReset = () => {
    setSelectedFile(null);
    setUploadError(null);
    setTaskId(null);
    setJobStatus(null);
    setIsUploading(false);
  };

  // Handle download
  const handleDownload = () => {
    if (taskId) {
      const downloadUrl = VideoAPI.getKeypointsDownloadUrl(taskId);
      window.open(downloadUrl, '_blank');
    }
  };

  // UI states
  const isProcessing = jobStatus && ['queued', 'processing'].includes(jobStatus.status);
  const isCompleted = jobStatus && jobStatus.status === 'completed';
  const isFailed = jobStatus && jobStatus.status === 'failed';

  return (
    <div className="keypoints-upload">
      {!taskId && (
        <div className="keypoints-upload-section">
          <div className="keypoints-header">
            <h2>3D Keypoints Extraction</h2>
            <p>Upload a video to extract 3D keypoints and download as .npy file</p>
          </div>

          <FileUpload
            selectedFile={selectedFile}
            onFileSelect={handleFileSelect}
            uploadError={uploadError}
            disabled={isUploading}
          />
          
          <div className="keypoints-controls">
            <div className="model-selection">
              <label htmlFor="keypoints-model-size">Model Size:</label>
              <select 
                id="keypoints-model-size" 
                value={modelSize} 
                onChange={e => setModelSize(e.target.value)} 
                disabled={isUploading}
              >
                <option value="xs">Small/Fast</option>
                <option value="s">Large/Slow</option>
              </select>
            </div>
            
            <button 
              className="keypoints-upload-btn" 
              onClick={handleUpload} 
              disabled={!selectedFile || isUploading}
            >
              <span className="btn-icon">🎯</span>
              {isUploading ? <span className="btn-spinner" /> : 'Extract 3D Keypoints'}
            </button>
          </div>

          {uploadError && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {uploadError}
            </div>
          )}
        </div>
      )}

      {isProcessing && (
        <div className="processing-section">
          <ProgressBar 
            status={jobStatus.status} 
            progress={jobStatus.progress} 
            message="Extracting 3D keypoints from video..."
          />
          <div className="processing-info">
            <p className="job-id">Job ID: {taskId}</p>
          </div>
        </div>
      )}

      {isCompleted && (
        <div className="results-section">
          <div className="success-content">
            <span className="success-icon">✅</span>
            <h3>3D Keypoints Extracted Successfully!</h3>
            <p>Your 3D keypoints have been extracted and are ready for download.</p>
            
            <div className="download-actions">
              <button className="download-btn" onClick={handleDownload}>
                <span className="btn-icon">⬇️</span>
                Download 3D Keypoints (.npy)
              </button>
              <button className="new-extraction-btn" onClick={handleReset}>
                Extract New Video
              </button>
            </div>
          </div>
        </div>
      )}

      {isFailed && (
        <div className="error-section">
          <div className="error-content">
            <span className="error-icon">❌</span>
            <h3>Extraction Failed</h3>
            <p>{jobStatus.error || 'An error occurred during 3D keypoints extraction.'}</p>
            <button className="retry-btn" onClick={handleReset}>Try Again</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default KeypointsUpload;