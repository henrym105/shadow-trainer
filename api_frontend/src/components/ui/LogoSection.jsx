import React from 'react';
import { useNavigate } from 'react-router-dom';

function LogoSection() {
  const navigate = useNavigate();

  return (
    <div
      className="logo-section"
      onClick={() => navigate('/')}
      style={{ cursor: 'pointer' }}
    >
      <img 
        src="/Shadow Trainer Logo Only.png" 
        alt="Shadow Trainer Logo Only" 
        className="logo"
      />
      <div className="logo-text">
        <h1>Shadow Trainer</h1>
        <p>AI-Powered 3D Motion Analysis</p>
      </div>
    </div>
  );
}

export default LogoSection;