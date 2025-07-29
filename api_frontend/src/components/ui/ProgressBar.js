import React from 'react';
import '../../styles/ProgressBar.css';
const STATUS_MAP = {
  queued: { color: '#ffa500', icon: '⏰', label: 'Queued' },
  processing: { color: '#4CAF50', icon: '⚙️', label: 'Processing' },
  completed: { color: '#2196F3', icon: '✅', label: 'Completed' },
  failed: { color: '#f44336', icon: '❌', label: 'Failed' }
};


const ProgressBar = ({ status, progress, proPlayerName }) => {
  const statusInfo = STATUS_MAP[status] || STATUS_MAP['queued'];
  
  const getProgressMessage = () => {
    if (status === 'processing' && proPlayerName) {
      return `Rendering ${proPlayerName} as your Shadow...`;
    }
    if (status === 'processing') {
      return 'Your video is being analyzed. This may take a few minutes.';
    }
    if (status === 'queued') return 'Waiting for available worker...';
    if (status === 'completed') return 'Processing complete!';
    if (status === 'failed') return 'There was an error during processing.';
    return '';
  };

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
        {getProgressMessage()}
      </div>
    </div>
  );
};

export default ProgressBar;
