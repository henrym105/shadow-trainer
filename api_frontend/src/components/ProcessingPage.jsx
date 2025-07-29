import React, { useState, useEffect } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import ProgressBar from './ProgressBar';
import LogoSection from './LogoSection';
import VideoAPI, { useJobPolling } from '../services/videoApi';

function ProcessingPage() {
  const { taskId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  
  const [jobStatus, setJobStatus] = useState({ status: 'queued', progress: 0 });
  
  // Get data passed from previous page
  const { videoFormat, selectedProFile, proOptions } = location.state || {};

  // Poll job status
  useJobPolling(taskId, setJobStatus, 2000);

  // UI states
  const isProcessing = jobStatus && ['queued', 'processing'].includes(jobStatus.status);
  const isCompleted = jobStatus && jobStatus.status === 'completed';
  const isFailed = jobStatus && jobStatus.status === 'failed';
  const isTerminated = jobStatus && jobStatus.status === 'terminated';

  // Redirect when job completes
  useEffect(() => {
    if (isCompleted) {
      if (videoFormat === 'dynamic_3d_animation') {
        navigate(`/visualizer/${taskId}`);
      } else {
        navigate(`/results/${taskId}`, { 
          state: { 
            taskId, 
            jobStatus, 
            videoFormat,
            selectedProFile,
            proOptions 
          } 
        });
      }
    }
  }, [isCompleted, videoFormat, taskId, navigate, jobStatus, selectedProFile, proOptions]);

  // Handle task termination
  const handleTerminate = async () => {
    if (!taskId) return;
    try {
      await VideoAPI.terminateTask(taskId);
      setJobStatus({ status: 'terminated', message: 'Task terminated by user' });
      setTimeout(() => {
        navigate('/');
      }, 2000);
    } catch (error) {
      console.error('Failed to terminate task:', error);
    }
  };

  // Get professional player name from selected file
  const getProPlayerName = () => {
    if (!proOptions || !selectedProFile) return 'Professional Player';
    const proOption = proOptions.find(opt => opt.filename === selectedProFile);
    return proOption?.name || selectedProFile.replace('_median.npy', '').replace(/([A-Z])/g, ' $1').trim();
  };

  return (
    <div className="app">
      <div className="app-container">
        <header className="app-header">
          <LogoSection />
        </header>
        <main className="app-main">
          {isProcessing && (
            <section className="processing-section">
              <ProgressBar 
                status={jobStatus.status} 
                progress={jobStatus.progress} 
                proPlayerName={getProPlayerName()} 
              />
              <div className="processing-info">
                <p className="job-id">Job ID: {taskId}</p>
                <button className="terminate-btn" onClick={handleTerminate}>
                  <span className="btn-icon">🛑</span>
                  Terminate Processing
                </button>
              </div>
            </section>
          )}
          
          {isFailed && (
            <section className="error-section">
              <div className="error-content">
                <span className="error-icon">❌</span>
                <h3>Processing Failed</h3>
                <p>{jobStatus.error || 'An error occurred during processing.'}</p>
                <button className="retry-btn" onClick={() => navigate('/')}>Try Again</button>
              </div>
            </section>
          )}
          
          {isTerminated && (
            <section className="terminated-section">
              <div className="terminated-content">
                <span className="terminated-icon">🛑</span>
                <h3>Processing Terminated</h3>
                <p>{jobStatus.message || 'Task was terminated by user request.'}</p>
                <button className="retry-btn" onClick={() => navigate('/')}>Start New Processing</button>
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}

export default ProcessingPage;