import React from 'react';

function RecordingTipsPopup({ isVisible, onClose }) {
  if (!isVisible) return null;

  return (
    <>
      <div className="popup-overlay" onClick={onClose}></div>
      <div className="recording-tips-popup">
        <div className="popup-content">
          <button
            className="popup-close"
            onClick={onClose}
          >
            ✕
          </button>
          <p className="popup-description">To get the best results from your video:</p>
          <ul className="popup-tips-list">
            <li>Try to be the only person in the video</li>
            <li>Keep the video short—just enough to capture your full pitch</li>
            <li>Stabilize camera by placing it on a stationary object</li>
            <li>Throw to the side with the camera facing your chest</li>
            <li>Keep your entire body in view, including arms and legs</li>
          </ul>
        </div>
      </div>
    </>
  );
}

export default RecordingTipsPopup;