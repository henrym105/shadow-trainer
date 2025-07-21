import React, { useState } from 'react';
import './VideoResult.css';

const VideoResult = ({ taskId, originalFilename, previewUrl, downloadUrl }) => {
  const [videoError, setVideoError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const handleVideoLoad = () => {
    setIsLoading(false);
    setVideoError(false);
  };

  const handleVideoError = () => {
    setIsLoading(false);
    setVideoError(true);
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = `/api/videos/${taskId}/download`;
    link.download = `processed_${originalFilename}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const copyLinkToClipboard = async () => {
    const fullUrl = `/api/videos/${taskId}/download`;
    try {
      await navigator.clipboard.writeText(fullUrl);
      alert('Download link copied to clipboard!');
    } catch (err) {
      console.error('Failed to copy link:', err);
      const textArea = document.createElement('textarea');
      textArea.value = fullUrl;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      alert('Download link copied to clipboard!');
    }
  };

  return (
    <div className="video-result-container">
      <div className="result-header">
        <h3>🎉 Your Shadow is Ready!</h3>
        <p>Motion analysis completed for: <strong>{originalFilename}</strong></p>
      </div>

      <div className="video-preview-section">
        {isLoading && (
          <div className="video-loading">
            <div className="loading-spinner"></div>
            <p>Loading preview...</p>
          </div>
        )}
        {videoError ? (
          <div className="video-error">
            <div className="error-icon">⚠️</div>
            <p>Unable to load video preview</p>
            <p className="error-message">But you can still download the processed video below</p>
          </div>
        ) : (
          <video
            controls
            className="result-video"
            onLoadedData={handleVideoLoad}
            onError={handleVideoError}
            onLoadStart={() => setIsLoading(true)}
            poster="/assets/video-poster.png"
            src={`/api/videos/${taskId}/preview`}
          >
            <source src={`/api/videos/${taskId}/preview`} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        )}
      </div>

      <div className="action-buttons">
        <button className="download-btn primary" onClick={handleDownload}>
          <span className="btn-icon">⬇️</span>
          Download Video
        </button>
        <button className="copy-link-btn secondary" onClick={copyLinkToClipboard}>
          <span className="btn-icon">🔗</span>
          Copy Link
        </button>
      </div>

      <div className="result-info">
        <div className="info-item">
          <span className="info-label">Job ID:</span>
          <span className="info-value">{taskId}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Processing:</span>
          <span className="info-value">2D/3D Pose Estimation Complete</span>
        </div>
      </div>
    </div>
  );
};

export default VideoResult;