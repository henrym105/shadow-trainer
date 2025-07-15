import React, { useEffect, useState } from "react";

function ProKeypointsSelector({ onSelect, disabled }) {
  const [proFiles, setProFiles] = useState([]);
  const [selected, setSelected] = useState("");

  useEffect(() => {
    fetch("/api/pro_keypoints/list")
      .then(res => res.json())
      .then(data => setProFiles(data.files || []));
  }, []);

  const handleChange = (e) => {
    setSelected(e.target.value);
    if (onSelect) onSelect(e.target.value);
  };

  return (
    <div className="model-selection pro-keypoints-selector">
      <label htmlFor="pro-player-select">Choose a professional player:</label>
      <select
        id="pro-player-select"
        value={selected}
        onChange={handleChange}
        disabled={disabled}
      >
        <option value="">Default (SnellBlake)</option>
        {proFiles.map(f => (
          <option key={f} value={f}>{f.replace(".npy", "")}</option>
        ))}
      </select>
    </div>
  );
}

export default ProKeypointsSelector;
