import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import FileUpload from '../ui/FileUpload';
import ProKeypointsSelector from '../ui/ProKeypointsSelector';
import KeypointsUpload from '../ui/KeypointsUpload';
import LogoSection from '../ui/LogoSection';
import RecordingTipsPopup from '../ui/RecordingTipsPopup';
import GitHubFooter from '../ui/GitHubFooter';
import VideoAPI from '../../services/videoApi';
import '../../styles/GitHubFooter.css';

const MODEL_SIZES = [
  { value: 'xs', label: 'Small/Fast' },
  { value: 's', label: 'Large/Slow' }
];

function AppPage() {
  const navigate = useNavigate();
  
  // State variables
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [modelSize, setModelSize] = useState('xs');
  const [isLefty, setIsLefty] = useState(false);
  const [selectedProFile, setSelectedProFile] = useState('DeanKremer_median.npy');
  const [proOptions, setProOptions] = useState([]);
  const [videoFormat, setVideoFormat] = useState('dynamic_3d_animation');
  const [activeTab, setActiveTab] = useState('video-processing');
  const [showRecordingTips, setShowRecordingTips] = useState(false);

  // Load pro keypoints options
  useEffect(() => {
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
      // Navigate to processing page with task data
      navigate(`/processing/${res.task_id}`, { 
        state: { 
          taskId: res.task_id, 
          videoFormat,
          selectedProFile,
          proOptions 
        } 
      });
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
      // Navigate to processing page with task data
      navigate(`/processing/${res.task_id}`, { 
        state: { 
          taskId: res.task_id, 
          videoFormat,
          selectedProFile,
          proOptions 
        } 
      });
      setSelectedFile(null);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="app">
      <RecordingTipsPopup 
        isVisible={showRecordingTips}
        onClose={() => setShowRecordingTips(false)}
      />
      <div className="app-container">
        <header className="app-header">
          <LogoSection />
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
            <section className="upload-section">
              <button className="sample-video-btn" onClick={handleSampleVideo} disabled={isUploading}>
                {isUploading ? <span className="btn-spinner" /> : 'Use Sample Video'}
              </button>
              <div className="sample-video-description" style={{ textAlign: 'center' }}>Try our sample video to see Shadow Trainer in action!</div>
              <div className="divider"><span>or</span></div>
              <div className="section-header">
                <h2>Upload Your Training Video</h2>
                <p>Get detailed motion analysis and pose estimation for your athletic performance</p>
                <div className="recording-tips-container">
                  <div className="recording-tips-button">
                    <button 
                      className="recording-tips-btn" 
                      onClick={() => setShowRecordingTips(true)}
                    >
                      📹 Recording Tips
                    </button>
                  </div>
                </div>
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
                          <span className={`toggle-label${isLefty ? ' active' : ''}`}>Left-Handed</span>
                          <label className="toggle-switch">
                            <input type="checkbox" checked={!isLefty} onChange={e => setIsLefty(!e.target.checked)} disabled={isUploading} />
                            <span className="slider"></span>
                          </label>
                          <span className={`toggle-label${!isLefty ? ' active' : ''}`}>Right-Handed</span>
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
                          <option value="dynamic_3d_animation">Dynamic 3D Animation</option>
                          <option value="combined">2D + 3D Side by Side</option>
                          <option value="3d_only">3D Skeleton Only</option>
                        </select>
                      </div>
                    </div>
                  </div>
                </div>
                <button className="upload-btn" onClick={handleUpload} disabled={!selectedFile || isUploading}>
                  <span className="btn-icon">⬆️</span>
                  {isUploading ? <span className="btn-spinner" /> : 'Create Your Shadow!'}
                </button>
              </div>
              {uploadError && (
                <div className="error-message">
                  <span className="error-icon">⚠️</span>
                  {uploadError}
                </div>
              )}
            </section>
          )}

          {activeTab === '3d-keypoints' && (
            <KeypointsUpload />
          )}
        </main>
        <GitHubFooter />
      </div>
    </div>
  );
}

export default AppPage;