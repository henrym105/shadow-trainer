/**
 * Progress Bar Component
 * Shows processing progress with status messages
 */

import React from 'react';
import './ProgressBar.css';

const ProgressBar = ({ 
  progress = 0, 
  status = 'queued', 
  message = '', 
  showPercentage = true,
  animated = true 
}) => {
  // Determine status color and icon
  const getStatusInfo = (status) => {
    switch (status) {
      case 'queued':
        return { color: '#ffa500', icon: '⏳', text: 'Queued' };
      case 'processing':
        return { color: '#4CAF50', icon: '⚙️', text: 'Processing' };
      case 'completed':
        return { color: '#2196F3', icon: '✅', text: 'Completed' };
      case 'failed':
        return { color: '#f44336', icon: '❌', text: 'Failed' };
      default:
        return { color: '#ccc', icon: '⏸️', text: 'Unknown' };
    }
  };

  const statusInfo = getStatusInfo(status);
  const clampedProgress = Math.max(0, Math.min(100, progress));

  return (
    <div className="progress-container">
      <div className="progress-header">
        <div className="status-info">
          <span className="status-icon">{statusInfo.icon}</span>
          <span className="status-text">{statusInfo.text}</span>
        </div>
        {showPercentage && (
          <span className="progress-percentage">
            {clampedProgress}%
          </span>
        )}
      </div>
      
      <div className="progress-bar-container">
        <div 
          className={`progress-bar ${animated ? 'animated' : ''}`}
          style={{ 
            width: `${clampedProgress}%`,
            backgroundColor: statusInfo.color
          }}
        >
          {animated && status === 'processing' && (
            <div className="progress-bar-shimmer"></div>
          )}
        </div>
      </div>
      
      {message && (
        <div className="progress-message">
          {message}
        </div>
      )}
    </div>
  );
};

export default ProgressBar;
