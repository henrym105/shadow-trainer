import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import SkeletonViewer from './Skeletonviewer';
import LogoSection from './LogoSection';
import './VisualizerPage.css';

function VisualizerPage() {
  const { taskId } = useParams();
  const navigate = useNavigate();
  const [userKeypoints, setUserKeypoints] = useState(null);
  const [proKeypoints, setProKeypoints] = useState(null);
  const [taskInfo, setTaskInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
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

  return (
    <div className="app" style={{ width: '100vw', maxWidth: 'none' }}>
      <div style={{ width: '95vw', margin: '0 auto', padding: '2rem' }}>
        {/* Header */}
        <header className="app-header">
          <LogoSection />
        </header>
        
        {/* Main Content Area - Animation + Controls Side by Side */}
        <div className="main-content-container">
          {/* Visualization Box */}
          <div className="visualization-box">
            {userKeypoints && proKeypoints ? (
              <>
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
                />
                
                {/* Skeleton Legend - Bottom Single Row */}
                <div style={{
                  position: 'absolute',
                  bottom: '20px',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  background: 'rgba(255, 255, 255, 0.9)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '8px',
                  padding: '15px 30px',
                  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  zIndex: 100,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '30px'
                }}>
                  <h4 style={{
                    margin: 0,
                    fontSize: '14px',
                    fontWeight: '600',
                    color: '#333'
                  }}>Skeleton Legend:</h4>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    fontSize: '12px',
                    color: '#555'
                  }}>
                    <div style={{
                      width: '16px',
                      height: '16px',
                      borderRadius: '50%',
                      border: '2px solid rgba(255, 255, 255, 0.8)',
                      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
                      background: '#ff4444'
                    }}></div>
                    <span>Your Movement</span>
                  </div>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    fontSize: '12px',
                    color: '#555'
                  }}>
                    <div style={{
                      width: '16px',
                      height: '16px',
                      borderRadius: '50%',
                      border: '2px solid rgba(255, 255, 255, 0.8)',
                      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
                      background: '#888888'
                    }}></div>
                    <span>{taskInfo?.pro_name || 'Professional Reference'}</span>
                  </div>
                </div>
              </>
            ) : (
              <div style={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center', 
                height: '100%' 
              }}>
                <div>Loading keypoints data...</div>
              </div>
            )}
          </div>
          
          {/* Visualization Controls Section - Now on the side */}
          <div className="controls-container">
            <h3 style={{
              color: '#333',
              fontSize: '1.3rem',
              marginBottom: '1.5rem',
              textAlign: 'center',
              fontWeight: '600',
              position: 'relative'
            }}>Visualization Controls</h3>
            
            {/* Control Groups */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              {/* Auto Rotation Control */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <label style={{ fontWeight: '600', color: '#333' }}>Auto Rotation:</label>
                <button 
                  onClick={handleTurntableToggle}
                  style={{
                    padding: '0.75rem 1.5rem',
                    background: turntable ? '#f44336' : '#2196F3',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '1rem',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    width: '100%'
                  }}
                >
                  {turntable ? 'Stop Rotation' : 'Start Rotation'}
                </button>
              </div>

              {/* Skeleton Visibility Controls */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', textAlign: 'center' }}>
                  <label style={{ fontWeight: '600', color: '#333' }}>Show User Skeleton:</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', justifyContent: 'center' }}>
                    <span style={{ fontWeight: '500', color: !showUserSkeleton ? '#4CAF50' : '#666', fontSize: '0.9rem' }}>Hidden</span>
                    <label style={{ position: 'relative', display: 'inline-block', width: '50px', height: '24px' }}>
                      <input 
                        type="checkbox" 
                        checked={showUserSkeleton} 
                        onChange={handleUserSkeletonToggle}
                        style={{ opacity: 0, width: 0, height: 0 }}
                      />
                      <span style={{
                        position: 'absolute',
                        cursor: 'pointer',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        backgroundColor: showUserSkeleton ? '#4CAF50' : '#ccc',
                        transition: '.4s',
                        borderRadius: '24px'
                      }}>
                        <span style={{
                          position: 'absolute',
                          height: '18px',
                          width: '18px',
                          left: showUserSkeleton ? '29px' : '3px',
                          bottom: '3px',
                          backgroundColor: 'white',
                          transition: '.4s',
                          borderRadius: '50%'
                        }}></span>
                      </span>
                    </label>
                    <span style={{ fontWeight: '500', color: showUserSkeleton ? '#4CAF50' : '#666', fontSize: '0.9rem' }}>Visible</span>
                  </div>
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', textAlign: 'center' }}>
                  <label style={{ fontWeight: '600', color: '#333' }}>Show Pro Skeleton:</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', justifyContent: 'center' }}>
                    <span style={{ fontWeight: '500', color: !showProSkeleton ? '#4CAF50' : '#666', fontSize: '0.9rem' }}>Hidden</span>
                    <label style={{ position: 'relative', display: 'inline-block', width: '50px', height: '24px' }}>
                      <input 
                        type="checkbox" 
                        checked={showProSkeleton} 
                        onChange={handleProSkeletonToggle}
                        style={{ opacity: 0, width: 0, height: 0 }}
                      />
                      <span style={{
                        position: 'absolute',
                        cursor: 'pointer',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        backgroundColor: showProSkeleton ? '#4CAF50' : '#ccc',
                        transition: '.4s',
                        borderRadius: '24px'
                      }}>
                        <span style={{
                          position: 'absolute',
                          height: '18px',
                          width: '18px',
                          left: showProSkeleton ? '29px' : '3px',
                          bottom: '3px',
                          backgroundColor: 'white',
                          transition: '.4s',
                          borderRadius: '50%'
                        }}></span>
                      </span>
                    </label>
                    <span style={{ fontWeight: '500', color: showProSkeleton ? '#4CAF50' : '#666', fontSize: '0.9rem' }}>Visible</span>
                  </div>
                </div>
              </div>

              {/* Playback Speed Control */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <label style={{ fontWeight: '600', color: '#333' }}>Playback Speed:</label>
                <select 
                  value={playbackSpeed} 
                  onChange={handleSpeedChange}
                  style={{
                    padding: '0.75rem',
                    border: '2px solid #e0e0e0',
                    borderRadius: '8px',
                    fontSize: '1rem',
                    background: 'white',
                    cursor: 'pointer',
                    transition: 'border-color 0.3s',
                    width: '100%'
                  }}
                >
                  <option value={0.25}>0.25x</option>
                  <option value={0.5}>0.5x</option>
                  <option value={1}>1x (Normal)</option>
                  <option value={1.5}>1.5x</option>
                  <option value={2}>2x</option>
                </select>
              </div>

              {/* Frame Control */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <label style={{ fontWeight: '600', color: '#333', textAlign: 'center' }}>Frame: {frame + 1} / {totalFrames}</label>
                <input
                  type="range"
                  min={0}
                  max={totalFrames - 1}
                  value={frame}
                  onChange={handleFrameChange}
                  style={{ 
                    width: '100%',
                    height: '6px',
                    borderRadius: '5px',
                    background: '#ddd',
                    outline: 'none'
                  }}
                />
              </div>

              {/* Playback Control */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <label style={{ fontWeight: '600', color: '#333' }}>Playback:</label>
                <button 
                  onClick={handlePlayPause}
                  style={{
                    padding: '0.75rem 1.5rem',
                    background: playing ? '#f44336' : '#4CAF50',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '1rem',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    width: '100%'
                  }}
                >
                  {playing ? 'Pause' : 'Play'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VisualizerPage;