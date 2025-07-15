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
      <label htmlFor="pro-player-select">Choose a professional player to shadow:</label>
      <select
        id="pro-player-select"
        value={selected}
        onChange={handleChange}
        disabled={disabled}
      >
        {proFiles.map(f => {
          // Remove .npy, split into Lastname and Firstname, and display as "Firstname Lastname"
          const name = f.replace('.npy', '');
          let displayName = name;
          if (name.match(/^[A-Za-z]+[A-Z][a-z]+$/)) {
            // Try to split at the transition from lowercase to uppercase
            const splitIdx = name.search(/[A-Z][a-z]+$/);
            if (splitIdx > 0) {
              const last = name.slice(0, splitIdx);
              const first = name.slice(splitIdx);
              displayName = `${first} ${last}`;
            }
          }
          return (
            <option key={f} value={f}>{displayName}</option>
          );
        })}
      </select>
    </div>
  );
}

export default ProKeypointsSelector;
