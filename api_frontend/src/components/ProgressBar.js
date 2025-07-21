import React from 'react';
import './ProgressBar.css';
const STATUS_MAP = {
  queued: { color: '#ffa500', icon: '⏰', label: 'Queued' },
  processing: { color: '#4CAF50', icon: '⚙️', label: 'Processing' },
  completed: { color: '#2196F3', icon: '✅', label: 'Completed' },
  failed: { color: '#f44336', icon: '❌', label: 'Failed' }
};


const ProgressBar = ({ status, progress }) => {
  const statusInfo = STATUS_MAP[status] || STATUS_MAP['queued'];
  return (
    <div className="progress-bar-card">
      <div className="progress-status" style={{ color: statusInfo.color }}>
        <span className="status-icon">{statusInfo.icon}</span>
        <span className="status-label">{statusInfo.label}</span>
      </div>
      <div className="progress-bar">
        <div
          className="progress-bar-fill"
          style={{ width: `${progress || 0}%`, background: statusInfo.color }}
        />
      </div>
      <div className="progress-percent">{progress ? `${progress}%` : '...'}</div>
      <div className="progress-desc">
        {status === 'processing' && 'Your video is being analyzed. This may take a few minutes.'}
        {status === 'queued' && 'Waiting for available worker...'}
        {status === 'completed' && 'Processing complete!'}
        {status === 'failed' && 'There was an error during processing.'}
      </div>
    </div>
  );
};

export default ProgressBar;
