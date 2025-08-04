import React from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import VideoResult from '../ui/VideoResult';
import LogoSection from '../ui/LogoSection';
import GitHubFooter from '../ui/GitHubFooter';
import '../../styles/GitHubFooter.css';

function ResultsPage() {
  const { taskId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  
  // Get data passed from processing page
  const { jobStatus, videoFormat } = location.state || {};

  // Handle reset - go back to app page
  const handleReset = () => {
    navigate('/app');
  };

  // If no job status data, redirect to app
  if (!jobStatus) {
    navigate('/app');
    return null;
  }

  return (
    <div className="app">
      <div className="app-container">
        <header className="app-header">
          <LogoSection />
        </header>
        <main className="app-main">
          <VideoResult
            taskId={taskId}
            jobStatus={jobStatus}
            onReset={handleReset}
          />
        </main>
        <GitHubFooter />
      </div>
    </div>
  );
}

export default ResultsPage;