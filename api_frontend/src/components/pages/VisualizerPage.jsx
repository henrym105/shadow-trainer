import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import SkeletonViewer from '../ui/Skeletonviewer';
import LogoSection from '../ui/LogoSection';
import PlotViewer from '../ui/PlotViewer';
import '../../styles/VisualizerPage.css';

function VisualizerPage() {
  const { taskId } = useParams();
  const navigate = useNavigate();
  const [userKeypoints, setUserKeypoints] = useState(null);
  const [proKeypoints, setProKeypoints] = useState(null);
  const [taskInfo, setTaskInfo] = useState(null);
  const [jointEvaluation, setJointEvaluation] = useState(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationTaskId, setEvaluationTaskId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showPlots, setShowPlots] = useState(false);
  const [hasMotionFeedback, setHasMotionFeedback] = useState(false);
  
  // Controls state
  const [playing, setPlaying] = useState(true);
  const [showUserSkeleton, setShowUserSkeleton] = useState(true);
  const [showProSkeleton, setShowProSkeleton] = useState(true);
  const [frame, setFrame] = useState(0);
  const [turntable, setTurntable] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  useEffect(() => {
    const fetchKeypoints = async () => {
      try {
        const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
        const [userRes, proRes, infoRes] = await Promise.all([
          fetch(`${apiBaseUrl}/videos/${taskId}/keypoints/user?format=flattened`),
          fetch(`${apiBaseUrl}/videos/${taskId}/keypoints/pro?format=flattened`),
          fetch(`${apiBaseUrl}/videos/${taskId}/info`)
        ]);
        
        if (userRes.ok && proRes.ok) {
          const userData = await userRes.json();
          const proData = await proRes.json();
          setUserKeypoints(userData.keypoints);
          setProKeypoints(proData.keypoints);
          
          // Get task info (pro name) - don't fail if this doesn't work
          if (infoRes.ok) {
            const infoData = await infoRes.json();
            setTaskInfo(infoData);
            
            // Check if motion_feedback already exists
            if (infoData.motion_feedback) {
              setJointEvaluation(infoData.motion_feedback);
              setHasMotionFeedback(true);
              setShowPlots(true);
            } else {
              setHasMotionFeedback(false);
            }
          }
        } else {
          setError('Failed to load keypoints data');
        }
      } catch (error) {
        console.error('Failed to fetch keypoints:', error);
        setError('Failed to load keypoints data');
      } finally {
        setLoading(false);
      }
    };

    if (taskId) {
      fetchKeypoints();
    }
  }, [taskId]);

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        flexDirection: 'column',
        fontFamily: 'Arial, sans-serif'
      }}>
        <div style={{ fontSize: '18px', marginBottom: '20px' }}>Loading 3D visualization...</div>
        <div style={{ fontSize: '14px', color: '#666' }}>Task ID: {taskId}</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        flexDirection: 'column',
        fontFamily: 'Arial, sans-serif'
      }}>
        <div style={{ fontSize: '18px', color: 'red', marginBottom: '20px' }}>Error: {error}</div>
        <div style={{ fontSize: '14px', color: '#666' }}>Task ID: {taskId}</div>
      </div>
    );
  }

  const totalFrames = userKeypoints?.length || 1;

  // Handle playback control
  const handlePlayPause = () => {
    setPlaying(!playing);
  };

  // Handle frame slider
  const handleFrameChange = (e) => {
    setPlaying(false);
    setFrame(Number(e.target.value));
  };

  // Handle playback speed
  const handleSpeedChange = (e) => {
    setPlaybackSpeed(Number(e.target.value));
  };

  // Handle skeleton visibility
  const handleUserSkeletonToggle = (e) => {
    setShowUserSkeleton(e.target.checked);
  };

  const handleProSkeletonToggle = (e) => {
    setShowProSkeleton(e.target.checked);
  };

  // Handle turntable rotation
  const handleTurntableToggle = () => {
    setTurntable(!turntable);
  };

  const handleGenerateEvaluation = async () => {
    if (!taskId) return;
    
    setEvaluationLoading(true);
    try {
      const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiBaseUrl}/videos/${taskId}/generate-evaluation`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const data = await response.json();
        setEvaluationTaskId(data.evaluation_task_id);
        
        // Poll for evaluation completion
        const pollEvaluation = setInterval(async () => {
          try {
            const statusResponse = await fetch(`${apiBaseUrl}/status/${data.evaluation_task_id}`);
            if (statusResponse.ok) {
              const statusData = await statusResponse.json();
              
              if (statusData.status === 'SUCCESS') {
                // Fetch updated info to get the motion feedback
                const infoResponse = await fetch(`${apiBaseUrl}/videos/${taskId}/info`);
                if (infoResponse.ok) {
                  const infoData = await infoResponse.json();
                  if (infoData.motion_feedback) {
                    setJointEvaluation(infoData.motion_feedback);
                    setHasMotionFeedback(true);
                  } else if (infoData.joint_evaluation_text) {
                    setJointEvaluation(infoData.joint_evaluation_text);
                    setHasMotionFeedback(true);
                  } else {
                    setJointEvaluation('No motion feedback available.');
                    console.log('No motion_feedback found, available keys:', Object.keys(infoData));
                  }
                }
                setShowPlots(true);
                setEvaluationLoading(false);
                clearInterval(pollEvaluation);
              } else if (statusData.status === 'FAILURE') {
                console.error('Evaluation failed:', statusData.error);
                setEvaluationLoading(false);
                clearInterval(pollEvaluation);
              }
            }
          } catch (error) {
            console.error('Error polling evaluation status:', error);
          }
        }, 2000);
        
        // Clean up polling after 5 minutes
        setTimeout(() => {
          clearInterval(pollEvaluation);
          setEvaluationLoading(false);
        }, 300000);
        
      } else {
        console.error('Failed to start evaluation generation');
        setEvaluationLoading(false);
      }
    } catch (error) {
      console.error('Error generating evaluation:', error);
      setEvaluationLoading(false);
    }
  };

  return (
    <div className="app" style={{ width: '100vw', maxWidth: 'none' }}>
      <div style={{ width: '95vw', margin: '0 auto', padding: '2rem' }}>
        {/* Header */}
        <header className="app-header">
          <LogoSection />
        </header>
        
        {/* Main Content: Visualization and Controls */}
        <div className="main-content-container">
          {/* 3D Skeleton Visualization */}
          <div className="visualization-box">
            {userKeypoints && proKeypoints ? (
              <>
                {/* Instructions */}
                <div className="user-instructions">
                  <div>Click and drag to rotate</div>
                  <div>Scroll or pinch to zoom</div>
                </div>
                
                <SkeletonViewer
                  keypointFrames={userKeypoints}
                  proKeypointFrames={proKeypoints}
                  playing={playing}
                  showUserSkeleton={showUserSkeleton}
                  showProSkeleton={showProSkeleton}
                  frame={frame}
                  turntable={turntable}
                  playbackSpeed={playbackSpeed}
                  onFrameChange={setFrame}
                  isLefty={taskInfo?.is_lefty || false}
                />
                
                {/* Legend */}
                <div className="skeleton-legend">
                  <div className="legend-item">
                    <div className="legend-color user"></div>
                    <span>You</span>
                  </div>
                  <div className="legend-item">
                    <div className="legend-color pro"></div>
                    <span>{taskInfo?.pro_name || 'Professional Reference'}</span>
                  </div>
                </div>
              </>
            ) : (
              <div className="centered-message">
                <div>Loading keypoints data...</div>
              </div>
            )}
          </div>
          
          {/* Controls */}
          <div className="controls-container">
            <h3 className="controls-title">Visualization Controls</h3>
            <div className="controls-content">
              {/* Auto Rotation */}
              <div className="control-group">
                <label className="control-label">Auto Rotation:</label>
                <button 
                  onClick={handleTurntableToggle}
                  className={`btn ${turntable ? 'btn-danger' : 'btn-secondary'}`}
                >
                  {turntable ? 'Stop Rotation' : 'Start Rotation'}
                </button>
              </div>

              {/* Skeleton Visibility */}
              <div className="flex flex-col gap-lg">
                <div className="control-group text-center">
                  <label className="control-label">User Skeleton:</label>
                  <div className="toggle-container">
                    <span className={`toggle-label ${!showUserSkeleton ? 'active' : ''}`}>Hidden</span>
                    <label className="toggle-switch">
                      <input 
                        type="checkbox" 
                        checked={showUserSkeleton} 
                        onChange={handleUserSkeletonToggle}
                      />
                      <span className="toggle-slider"></span>
                    </label>
                    <span className={`toggle-label ${showUserSkeleton ? 'active' : ''}`}>Visible</span>
                  </div>
                </div>
                
                <div className="control-group text-center">
                  <label className="control-label">Pro Skeleton:</label>
                  <div className="toggle-container">
                    <span className={`toggle-label ${!showProSkeleton ? 'active' : ''}`}>Hidden</span>
                    <label className="toggle-switch">
                      <input 
                        type="checkbox" 
                        checked={showProSkeleton} 
                        onChange={handleProSkeletonToggle}
                      />
                      <span className="toggle-slider"></span>
                    </label>
                    <span className={`toggle-label ${showProSkeleton ? 'active' : ''}`}>Visible</span>
                  </div>
                </div>
              </div>

              {/* Playback Speed */}
              <div className="control-group">
                <label className="control-label">Playback Speed:</label>
                <select 
                  value={playbackSpeed} 
                  onChange={handleSpeedChange}
                  className="form-select"
                >
                  <option value={0.25}>0.25x</option>
                  <option value={0.5}>0.5x</option>
                  <option value={1}>1x (Normal)</option>
                  <option value={1.5}>1.5x</option>
                  <option value={2}>2x</option>
                </select>
              </div>

              {/* Frame Slider */}
              <div className="control-group frame-control">
                <label className="control-label">
                  Time: {((frame + 1) / 30).toFixed(1)}s / {(totalFrames / 30).toFixed(1)}s
                </label>
                <input
                  type="range"
                  min={0}
                  max={totalFrames - 1}
                  value={frame}
                  onChange={handleFrameChange}
                  className="frame-slider"
                />
              </div>

              {/* Play/Pause */}
              <div className="control-group">
                <label className="control-label">Playback:</label>
                <button 
                  onClick={handlePlayPause}
                  className={`btn ${playing ? 'btn-danger' : 'btn-primary'}`}
                >
                  {playing ? 'Pause' : 'Play'}
                </button>
              </div>
            </div>
          </div>
        </div>
        
        {/* Feedback or Generate Analysis */}
        {!hasMotionFeedback && !evaluationLoading ? (
          <div className="visualization-box analysis-button-container">
            <button 
              onClick={handleGenerateEvaluation}
              disabled={evaluationLoading}
              className="analysis-button"
            >
              Generate Movement Analysis
            </button>
          </div>
        ) : (
          <div className="visualization-box analysis-button-container">
            <h3 className="evaluation-title">Motion Analysis Feedback</h3>
            {evaluationLoading ? (
              <div className="evaluation-loading">
                <div className="spinner spinner-large"></div>
                <p>Gathering personalized feedback from your AI baseball coach...</p>
              </div>
            ) : (
              <div className="evaluation-content">
                <ReactMarkdown>{jointEvaluation}</ReactMarkdown>
              </div>
            )}
          </div>
        )}
        
        {/* Motion Analysis Plots */}
        {showPlots && (
          <div className="visualization-box plots-container">
            <h3 className="plots-title">
              Compare your motion with {taskInfo?.pro_name || 'the MLB pro'}
            </h3>
            <div className="plots-grid">
              <PlotViewer 
                taskId={taskId} 
                plotType="hip_rotation" 
                title="Hip Rotation Analysis" 
              />
              <PlotViewer 
                taskId={taskId} 
                plotType="shoulder_rotation" 
                title="Shoulder Rotation Analysis" 
              />
              <PlotViewer 
                taskId={taskId} 
                plotType="hip_shoulder_separation" 
                title="Hip-Shoulder Separation" 
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default VisualizerPage;