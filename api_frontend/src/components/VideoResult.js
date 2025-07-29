import React, { useState } from 'react';
import VideoAPI from '../services/videoApi';
import './VideoResult.css';

const VideoResult = ({ taskId, jobStatus, onReset }) => {
  const [copySuccess, setCopySuccess] = useState(false);
  const downloadUrl = VideoAPI.getDownloadUrl(taskId);
  const previewUrl = VideoAPI.getPreviewUrl(taskId);

  const handleCopy = () => {
    navigator.clipboard.writeText(downloadUrl);
    setCopySuccess(true);
    setTimeout(() => setCopySuccess(false), 2000);
  };

  return (
    <div className="video-result-card">
      <h3>Processed Video</h3>
      <div className="video-player">
        <video src={previewUrl} controls width="400" height="300" />
      </div>
      <div className="result-actions">
        <a href={downloadUrl} download className="download-btn">Download Video</a>
        <button className="copy-btn" onClick={handleCopy}>
          {copySuccess ? 'Copied!' : 'Copy Download Link'}
        </button>
        <button className="reset-btn" onClick={onReset}>Process Another Video</button>
      </div>
      <div className="result-meta">
        <div><strong>Job ID:</strong> {taskId}</div>
        <div><strong>Status:</strong> {jobStatus.status}</div>
        <div><strong>File:</strong> {jobStatus.result?.original_filename || 'N/A'}</div>
        <div><strong>Size:</strong> {jobStatus.result?.file_size ? `${(jobStatus.result.file_size/1024/1024).toFixed(2)} MB` : 'N/A'}</div>
      </div>
    </div>
  );
};

export default VideoResult;
