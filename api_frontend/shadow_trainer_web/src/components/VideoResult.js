/**
 * Video Result Component
 * Displays processed video with download options
 */

import React, { useState, useEffect } from 'react';
import './VideoResult.css';
import ThreeJSkeleton from './ThreeJSkeleton';
import { VideoAPI } from '../services/videoApi';

const VideoResult = ({ jobId, originalFilename, previewUrl, downloadUrl, resultView = 'both' }) => {
  const [videoError, setVideoError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [userKeypoints, setUserKeypoints] = useState(null);
  const [proKeypoints, setProKeypoints] = useState(null);
  const [showUser, setShowUser] = useState(true);
  const [showPro, setShowPro] = useState(false);
  const [frameIdx, setFrameIdx] = useState(0);
  const [maxFrames, setMaxFrames] = useState(0);

  // Fetch keypoints on mount
  useEffect(() => {
    let isMounted = true;
    VideoAPI.getUser3DKeypoints(jobId).then(arr => {
      if (isMounted) {
        setUserKeypoints(arr);
        setMaxFrames(arr.length);
      }
    }).catch(() => setUserKeypoints(null));
    VideoAPI.getPro3DKeypoints(jobId).then(arr => {
      if (isMounted) setProKeypoints(arr);
    }).catch(() => setProKeypoints(null));
    return () => { isMounted = false; };
  }, [jobId]);

  const handleVideoLoad = () => {
    setIsLoading(false);
    setVideoError(false);
  };

  const handleVideoError = () => {
    setIsLoading(false);
    setVideoError(true);
  };

  const handleDownload = () => {
    // Create a temporary link to trigger download
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `processed_${originalFilename}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const copyLinkToClipboard = async () => {
    try {
      const fullUrl = `http://www.shadow-trainer.com${downloadUrl}`;
      await navigator.clipboard.writeText(fullUrl);
      // You could add a toast notification here
      alert('Download link copied to clipboard!');
    } catch (err) {
      console.error('Failed to copy link:', err);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = downloadUrl;
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

      {/* 3D Skeleton Viewer Section */}
      {(resultView === 'threejs' || resultView === 'both') && (
        <div className="skeleton-viewer-section">
          <div className="skeleton-toggles">
            <label>
              <input type="checkbox" checked={showUser} onChange={e => setShowUser(e.target.checked)} />
              Show User
            </label>
            <label>
              <input type="checkbox" checked={showPro} onChange={e => setShowPro(e.target.checked)} />
              Show Pro
            </label>
          </div>
          <div className="skeleton-frame-slider">
            <label>Frame: </label>
            <input
              type="range"
              min={0}
              max={Math.max(0, maxFrames - 1)}
              value={frameIdx}
              onChange={e => setFrameIdx(Number(e.target.value))}
              disabled={maxFrames === 0}
            />
            <span>{frameIdx + 1} / {maxFrames}</span>
          </div>
          <div className="skeleton-canvas">
            <ThreeJSkeleton
              userKeypoints={userKeypoints}
              proKeypoints={proKeypoints}
              showUser={showUser}
              showPro={showPro}
              frameIdx={frameIdx}
            />
          </div>
        </div>
      )}

      {/* Video Preview Section */}
      {(resultView === 'video' || resultView === 'both') && (
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
            >
              <source src={previewUrl} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          )}
        </div>
      )}

      <div className="action-buttons">
        <button 
          className="download-btn primary"
          onClick={handleDownload}
        >
          <span className="btn-icon">⬇️</span>
          Download Video
        </button>
        <button 
          className="copy-link-btn secondary"
          onClick={copyLinkToClipboard}
        >
          <span className="btn-icon">🔗</span>
          Copy Link
        </button>
      </div>

      <div className="result-info">
        <div className="info-item">
          <span className="info-label">Job ID:</span>
          <span className="info-value">{jobId}</span>
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
