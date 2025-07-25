import React from 'react';
import './ProKeypointsSelector.css';
import './ProKeypointsSelector.css';

const ProKeypointsSelector = ({ options, value, onChange, disabled }) => {
  return (
    <div className="pro-keypoints-selector">
      <label htmlFor="pro-keypoints" className="option-header">Professional Player:</label>
      <select
        id="pro-keypoints"
        value={value}
        onChange={e => onChange(e.target.value)}
        disabled={disabled}
      >
        {options.length === 0 && <option value="">Loading...</option>}
        {options.map(opt => (
          <option key={opt.filename} value={opt.filename}>
            {opt.name ? `${opt.name} (${opt.team})` : opt.filename}
          </option>
        ))}
      </select>
    </div>
  );
};

export default ProKeypointsSelector;
