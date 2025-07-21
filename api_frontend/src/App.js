import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import ProgressBar from './components/ProgressBar';
import VideoResult from './components/VideoResult';
import ProKeypointsSelector from './components/ProKeypointsSelector';
import VideoAPI, { useJobPolling } from './services/videoApi';
import './App.css';

const MODEL_SIZES = [
  { value: 'xs', label: 'Extra Small' },
  { value: 's', label: 'Small' },
  { value: 'm', label: 'Medium' },
  { value: 'l', label: 'Large' }
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
  const [selectedProFile, setSelectedProFile] = useState('Spencer_Strider.npy');
  const [proOptions, setProOptions] = useState([]);

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
      const res = await VideoAPI.uploadVideo(selectedFile, modelSize, isLefty, selectedProFile);
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
      const res = await VideoAPI.processSampleLeftyVideo(modelSize, selectedProFile);
      setTaskId(res.task_id);
      setJobStatus({ status: 'queued', progress: 0 });
      setSelectedFile(null);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setIsUploading(false);
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

  // UI states
  const isProcessing = jobStatus && ['queued', 'processing'].includes(jobStatus.status);
  const isCompleted = jobStatus && jobStatus.status === 'completed';
  const isFailed = jobStatus && jobStatus.status === 'failed';

  return (
    <div className="app-root">
      <header className="app-header">
        <img src="/Shadow Trainer Logo.png" alt="Shadow Trainer Logo" className="logo" />
        <div>
          <h1>Shadow Trainer</h1>
          <p>AI-Powered Motion Analysis</p>
        </div>
      </header>
      <main className="app-main">
        {!taskId && (
          <div className="upload-card">
            <h2>Upload Your Training Video</h2>
            <p>Get detailed motion analysis and pose estimation for your athletic performance</p>
            <FileUpload
              selectedFile={selectedFile}
              onFileSelect={handleFileSelect}
              uploadError={uploadError}
              disabled={isUploading}
            />
            <div className="config-row">
              <label>Model Size:</label>
              <select value={modelSize} onChange={e => setModelSize(e.target.value)} disabled={isUploading}>
                {MODEL_SIZES.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
              <label>
                <input type="checkbox" checked={isLefty} onChange={e => setIsLefty(e.target.checked)} disabled={isUploading} />
                Left-handed
              </label>
              <ProKeypointsSelector
                options={proOptions}
                value={selectedProFile}
                onChange={setSelectedProFile}
                disabled={isUploading}
              />
            </div>
            <div className="action-row">
              <button className="upload-btn" onClick={handleUpload} disabled={!selectedFile || isUploading}>Upload Video</button>
              <span className="or-divider">or</span>
              <button className="sample-btn" onClick={handleSampleVideo} disabled={isUploading}>
                🎯 Use Sample Video
              </button>
            </div>
            <div className="sample-desc">Try our sample left-handed baseball pitch for a quick demo</div>
          </div>
        )}
        {isProcessing && (
          <ProgressBar status={jobStatus.status} progress={jobStatus.progress} />
        )}
        {isCompleted && (
          <VideoResult
            taskId={taskId}
            jobStatus={jobStatus}
            onReset={handleReset}
          />
        )}
        {isFailed && (
          <div className="error-card">
            <h3>Processing Failed</h3>
            <p>{jobStatus.error || 'An error occurred during processing.'}</p>
            <button onClick={handleReset}>Try Again</button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
