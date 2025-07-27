import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import ProgressBar from './components/ProgressBar';
import VideoResult from './components/VideoResult';
import ProKeypointsSelector from './components/ProKeypointsSelector';
import KeypointsUpload from './components/KeypointsUpload';
import VideoAPI, { useJobPolling } from './services/videoApi';
import './App.css';

const MODEL_SIZES = [
  { value: 'xs', label: 'Small/Fast' },
  { value: 's', label: 'Large/Slow' }
];

function App() {
  // State variables
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [modelSize, setModelSize] = useState('xs');
  const [isLefty, setIsLefty] = useState(false);
  const [selectedProFile, setSelectedProFile] = useState('BlakeSnell_median.npy');
  const [proOptions, setProOptions] = useState([]);
  const [videoFormat, setVideoFormat] = useState('combined'); // 'combined' or '3d_only'
  const [activeTab, setActiveTab] = useState('video-processing'); // 'video-processing' or '3d-keypoints'

  // Poll job status
  useJobPolling(taskId, setJobStatus, 2000);

  // Load pro keypoints options
  React.useEffect(() => {
    VideoAPI.getProKeypointsList().then(setProOptions).catch(() => {});
  }, []);

  // Handle file selection
  const handleFileSelect = (file, error) => {
    setSelectedFile(file);
    setUploadError(error);
  };

  // Handle upload
  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setUploadError(null);
    try {
      const res = await VideoAPI.uploadVideo(selectedFile, modelSize, isLefty, selectedProFile, videoFormat);
      setTaskId(res.task_id);
      setJobStatus({ status: 'queued', progress: 0 });
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  // Handle sample video
  const handleSampleVideo = async () => {
    setIsUploading(true);
    setUploadError(null);
    try {
      const res = await VideoAPI.processSampleLeftyVideo(modelSize, selectedProFile, videoFormat);
      setTaskId(res.task_id);
      setJobStatus({ status: 'queued', progress: 0 });
      setSelectedFile(null);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  // Handle task termination
  const handleTerminate = async () => {
    if (!taskId) return;
    try {
      await VideoAPI.terminateTask(taskId);
      setJobStatus({ status: 'terminated', message: 'Task terminated by user' });
      setTimeout(() => {
        handleReset();
      }, 2000);
    } catch (error) {
      setUploadError(error.message);
    }
  };

  // Reset for new upload
  const handleReset = () => {
    setSelectedFile(null);
    setUploadError(null);
    setTaskId(null);
    setJobStatus(null);
    setIsUploading(false);
  };

  // Get professional player name from selected file
  const getProPlayerName = () => {
    const proOption = proOptions.find(opt => opt.filename === selectedProFile);
    return proOption?.name || selectedProFile.replace('_median.npy', '').replace(/([A-Z])/g, ' $1').trim();
  };

  // UI states
  const isProcessing = jobStatus && ['queued', 'processing'].includes(jobStatus.status);
  const isCompleted = jobStatus && jobStatus.status === 'completed';
  const isFailed = jobStatus && jobStatus.status === 'failed';
  const isTerminated = jobStatus && jobStatus.status === 'terminated';

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
          <div className="tab-navigation">
            <button 
              className={`tab-button ${activeTab === 'video-processing' ? 'active' : ''}`}
              onClick={() => setActiveTab('video-processing')}
            >
              Video Processing
            </button>
            <button 
              className={`tab-button ${activeTab === '3d-keypoints' ? 'active' : ''}`}
              onClick={() => setActiveTab('3d-keypoints')}
            >
              3D Keypoints Extraction
            </button>
          </div>

          {activeTab === 'video-processing' && (
            <>
              {!taskId && (
                <section className="upload-section">
              <div className="section-header">
                <h2>Upload Your Training Video</h2>
                <p>Get detailed motion analysis and pose estimation for your athletic performance</p>
              </div>
              <FileUpload
                selectedFile={selectedFile}
                onFileSelect={handleFileSelect}
                uploadError={uploadError}
                disabled={isUploading}
              />
              <div className="upload-controls">
                <div className="options-section">
                  <h3 className="options-title">Player Configuration</h3>
                  <div className="options-grid">
                    <div className="option-group">
                      <div className="throwing-style-selection">
                        <label className="option-header">Throwing Style:</label>
                        <div className="toggle-container">
                          <span className={`toggle-label${!isLefty ? ' active' : ''}`}>Right-Handed</span>
                          <label className="toggle-switch">
                            <input type="checkbox" checked={isLefty} onChange={e => setIsLefty(e.target.checked)} disabled={isUploading} />
                            <span className="slider"></span>
                          </label>
                          <span className={`toggle-label${isLefty ? ' active' : ''}`}>Left-Handed</span>
                        </div>
                      </div>
                      <ProKeypointsSelector
                        options={proOptions}
                        value={selectedProFile}
                        onChange={setSelectedProFile}
                        disabled={isUploading}
                      />
                    </div>
                  </div>
                </div>
                <div className="options-section">
                  <h3 className="options-title">Model & Output Settings</h3>
                  <div className="options-grid">
                    <div className="option-group">
                      <div className="model-selection">
                        <label htmlFor="model-size" className="option-header">Model Size:</label>
                        <select id="model-size" value={modelSize} onChange={e => setModelSize(e.target.value)} disabled={isUploading}>
                          {MODEL_SIZES.map(opt => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                          ))}
                        </select>
                      </div>
                      <div className="video-format-selection">
                        <label htmlFor="video-format" className="option-header">Output Format:</label>
                        <select id="video-format" value={videoFormat} onChange={e => setVideoFormat(e.target.value)} disabled={isUploading}>
                          <option value="combined">2D + 3D Side by Side</option>
                          <option value="3d_only">3D Skeleton Only</option>
                        </select>
                      </div>
                    </div>
                  </div>
                </div>
                <button className="upload-btn" onClick={handleUpload} disabled={!selectedFile || isUploading}>
                  <span className="btn-icon">⬆️</span>
                  {isUploading ? <span className="btn-spinner" /> : 'Upload Video'}
                </button>
                <div className="divider"><span>or</span></div>
                <button className="sample-video-btn" onClick={handleSampleVideo} disabled={isUploading}>
                  <span className="btn-icon">🎯</span>
                  {isUploading ? <span className="btn-spinner" /> : 'Use Sample Video'}
                </button>
                <div className="sample-video-description">Try our sample left-handed baseball pitch for a quick demo</div>
              </div>
              {uploadError && (
                <div className="error-message">
                  <span className="error-icon">⚠️</span>
                  {uploadError}
                </div>
              )}
            </section>
          )}
          {isProcessing && (
            <section className="processing-section">
              <ProgressBar status={jobStatus.status} progress={jobStatus.progress} proPlayerName={getProPlayerName()} />
              <div className="processing-info">
                <p className="job-id">Job ID: {taskId}</p>
                <button className="terminate-btn" onClick={handleTerminate}>
                  <span className="btn-icon">🛑</span>
                  Terminate Processing
                </button>
              </div>
            </section>
          )}
          {isCompleted && (
            <VideoResult
              taskId={taskId}
              jobStatus={jobStatus}
              onReset={handleReset}
            />
          )}
          {isFailed && (
            <section className="error-section">
              <div className="error-content">
                <span className="error-icon">❌</span>
                <h3>Processing Failed</h3>
                <p>{jobStatus.error || 'An error occurred during processing.'}</p>
                <button className="retry-btn" onClick={handleReset}>Try Again</button>
              </div>
            </section>
          )}
          {isTerminated && (
            <section className="terminated-section">
              <div className="terminated-content">
                <span className="terminated-icon">🛑</span>
                <h3>Processing Terminated</h3>
                <p>{jobStatus.message || 'Task was terminated by user request.'}</p>
                <button className="retry-btn" onClick={handleReset}>Start New Processing</button>
              </div>
            </section>
          )}
            </>
          )}

          {activeTab === '3d-keypoints' && (
            <KeypointsUpload />
          )}
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

export default App;
