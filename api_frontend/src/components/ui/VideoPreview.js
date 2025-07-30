import React from 'react';
import VideoAPI from '../../services/videoApi';

const VideoPreview = ({ 
  taskId, 
  videoUrl, 
  title = "Video Preview", 
  width = 400, 
  height = 300, 
  className = "" 
}) => {
  // Determine video source: either from taskId (server) or direct URL (local file)
  const videoSrc = taskId ? VideoAPI.getPreviewUrl(taskId) : videoUrl;
  
  if (!videoSrc) {
    return null;
  }

  return (
    <div className={`video-preview-section ${className}`}>
      <h3>{title}</h3>
      <div className="video-preview">
        <video 
          src={videoSrc} 
          controls 
          width={width} 
          height={height}
          style={{ borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
        />
      </div>
    </div>
  );
};

export default VideoPreview;