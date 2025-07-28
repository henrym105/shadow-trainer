import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import SkeletonViewer from './Skeletonviewer';

function VisualizerPage() {
  const { taskId } = useParams();
  const [userKeypoints, setUserKeypoints] = useState(null);
  const [proKeypoints, setProKeypoints] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchKeypoints = async () => {
      try {
        const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
        const [userRes, proRes] = await Promise.all([
          fetch(`${apiBaseUrl}/videos/${taskId}/keypoints/user?format=flattened`),
          fetch(`${apiBaseUrl}/videos/${taskId}/keypoints/pro?format=flattened`)
        ]);
        
        if (userRes.ok && proRes.ok) {
          const userData = await userRes.json();
          const proData = await proRes.json();
          setUserKeypoints(userData.keypoints);
          setProKeypoints(proData.keypoints);
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

  return (
    <div style={{ height: '100vh', position: 'relative' }}>
      <div style={{ 
        position: 'absolute', 
        top: '10px', 
        left: '50%', 
        transform: 'translateX(-50%)',
        zIndex: 100,
        background: 'rgba(255,255,255,0.9)', 
        padding: '10px 20px',
        borderRadius: '8px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        fontSize: '16px',
        fontWeight: 'bold',
        fontFamily: 'Arial, sans-serif'
      }}>
        Shadow Trainer - 3D Motion Visualization
      </div>
      {userKeypoints && proKeypoints ? (
        <SkeletonViewer
          keypointFrames={userKeypoints}
          proKeypointFrames={proKeypoints}
        />
      ) : (
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh' 
        }}>
          <div>Loading keypoints data...</div>
        </div>
      )}
    </div>
  );
}

export default VisualizerPage;