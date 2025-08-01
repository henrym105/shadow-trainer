import React, { useState } from 'react';
import VideoAPI from '../../services/videoApi';

const VideoPreview = ({ 
  taskId, 
  videoUrl, 
  title = "Video Preview", 
  width = 400, 
  height = 300, 
  className = "",
  showOriginal = false
}) => {
  const [videoError, setVideoError] = useState(false);
  const [loading, setLoading] = useState(true);

  // Determine video source: either from taskId (server) or direct URL (local file)
  const videoSrc = taskId 
    ? (showOriginal ? VideoAPI.getOriginalVideoUrl(taskId) : VideoAPI.getPreviewUrl(taskId))
    : videoUrl;
  
  if (!videoSrc) {
    return null;
  }

  const handleVideoError = (e) => {
    console.error('Video load error:', e.target.error);
    setVideoError(true);
    setLoading(false);
  };

  const handleVideoLoad = () => {
    setLoading(false);
    setVideoError(false);
  };

  return (
    <div className={`video-preview-section ${className}`}>
      <h3>{title}</h3>
      <div className="video-preview">
        {loading && !videoError && (
          <div style={{ 
            width, 
            height, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            backgroundColor: '#f0f0f0',
            borderRadius: '8px'
          }}>
            Loading video...
          </div>
        )}
        {videoError ? (
          <div style={{ 
            width, 
            height, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            backgroundColor: '#f8f8f8',
            border: '2px dashed #ccc',
            borderRadius: '8px',
            flexDirection: 'column'
          }}>
            <span style={{ fontSize: '24px', marginBottom: '8px' }}>📹</span>
            <span>Video not available</span>
            <small style={{ color: '#666', marginTop: '4px' }}>
              {showOriginal ? 'Original video' : 'Preview'} not ready
            </small>
          </div>
        ) : (
          <video 
            src={videoSrc} 
            controls 
            muted
            loop
            width={width} 
            height={height}
            style={{ borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
            onError={handleVideoError}
            onLoadedData={handleVideoLoad}
            onLoadStart={() => setLoading(true)}
          />
        )}
      </div>
    </div>
  );
};

export default VideoPreview;