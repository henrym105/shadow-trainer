import React from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import VideoResult from './VideoResult';

function ResultsPage() {
  const { taskId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  
  // Get data passed from processing page
  const { jobStatus, videoFormat } = location.state || {};

  // Handle reset - go back to home page
  const handleReset = () => {
    navigate('/');
  };

  // If no job status data, redirect to home
  if (!jobStatus) {
    navigate('/');
    return null;
  }

  return (
    <div className="app">
      <div className="app-container">
        <header className="app-header">
          <div className="logo-section">
            <img src="/Shadow Trainer Logo.png" alt="Shadow Trainer Logo" className="logo" />
            <div className="logo-text">
              <h1>Shadow Trainer</h1>
              <p>AI-Powered Motion Analysis</p>
            </div>
          </div>
        </header>
        <main className="app-main">
          <VideoResult
            taskId={taskId}
            jobStatus={jobStatus}
            onReset={handleReset}
          />
        </main>
        <footer>
          <div
            className="github-footer-link"
            style={{
              background: '#fff',
              borderRadius: '10px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.07)',
              padding: '18px 0',
              margin: '32px auto 0 auto',
              textAlign: 'center',
              maxWidth: '420px',
              fontSize: '1.08em',
              fontWeight: 500
            }}
          >
            <a
              href="https://github.com/henrym105/shadow-trainer/tree/develop"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: '#0366d6', textDecoration: 'underline' }}
            >
              View Source on GitHub
            </a>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default ResultsPage;