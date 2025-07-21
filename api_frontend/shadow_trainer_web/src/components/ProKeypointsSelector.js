import React, { useEffect, useState } from "react";

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

function ProKeypointsSelector({ onSelect, disabled }) {
  const [proFiles, setProFiles] = useState([]);
  const [selected, setSelected] = useState("");

  useEffect(() => {
    fetch(`${API_BASE_URL}/pro_keypoints/list`)
      .then(res => res.json())
      .then(data => setProFiles(data.files || []));
  }, []);

  const handleChange = (e) => {
    setSelected(e.target.value);
    if (onSelect) onSelect(e.target.value);
  };

  return (
    <div className="model-selection pro-keypoints-selector">
      <label htmlFor="pro-player-select">Choose a professional player to shadow:</label>
      <select
        id="pro-player-select"
        value={selected}
        onChange={handleChange}
        disabled={disabled}
      >
        {proFiles.map(f => (
          <option key={f.filename} value={f.filename}>
            {f.name} ({f.team})
          </option>
        ))}
      </select>
    </div>
  );
}

export default ProKeypointsSelector;
