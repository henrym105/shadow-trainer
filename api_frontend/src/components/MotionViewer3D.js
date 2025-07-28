import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';
import VideoAPI from '../services/videoApi';
import './MotionViewer3D.css';

// Joint connections based on backend inference.py show3Dpose function
const SKELETON_CONNECTIONS = [
  [0, 1], [0, 4], [1, 2], [2, 3], [4, 5], [5, 6], // legs
  [0, 7], [7, 8], [8, 9], [9, 10], // spine to head
  [8, 11], [11, 12], [12, 13], // right arm
  [8, 14], [14, 15], [15, 16] // left arm
];

// Joint names for reference (17 joints total)
const JOINT_NAMES = [
  "Hip", "Right Hip", "Right Knee", "Right Ankle",
  "Left Hip", "Left Knee", "Left Ankle", "Spine",
  "Thorax", "Neck", "Head", "Left Shoulder",
  "Left Elbow", "Left Wrist", "Right Shoulder",
  "Right Elbow", "Right Wrist"
];

// Enhanced skeleton component with improved visual quality
function Skeleton({ keypoints, color = "red", visible = true, opacity = 1.0 }) {
  const meshRef = useRef();
  
  const { bones, joints } = useMemo(() => {
    if (!keypoints || !visible) return { bones: [], joints: [] };
    
    const bones = [];
    const joints = [];
    
    // Create enhanced joints (spheres at each keypoint with better shading)
    keypoints.forEach((joint, index) => {
      const jointSize = index === 0 ? 0.035 : 0.025; // Larger hip joint
      joints.push(
        <mesh key={`joint-${index}`} position={[joint[0], joint[1], joint[2]]}>
          <sphereGeometry args={[jointSize, 16, 16]} />
          <meshStandardMaterial 
            color={color} 
            transparent 
            opacity={opacity}
            metalness={0.1}
            roughness={0.4}
          />
        </mesh>
      );
    });
    
    // Create enhanced bones (cylindrical shapes between connected joints)
    SKELETON_CONNECTIONS.forEach(([startIdx, endIdx], index) => {
      const start = keypoints[startIdx];
      const end = keypoints[endIdx];
      
      if (start && end) {
        const startVec = new THREE.Vector3(start[0], start[1], start[2]);
        const endVec = new THREE.Vector3(end[0], end[1], end[2]);
        const direction = new THREE.Vector3().subVectors(endVec, startVec);
        const length = direction.length();
        const midpoint = new THREE.Vector3().addVectors(startVec, endVec).multiplyScalar(0.5);
        
        // Create quaternion for rotation
        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
        
        bones.push(
          <mesh key={`bone-${index}`} position={midpoint.toArray()} quaternion={quaternion.toArray()}>
            <cylinderGeometry args={[0.008, 0.008, length, 8]} />
            <meshStandardMaterial 
              color={color} 
              transparent 
              opacity={opacity * 0.8}
              metalness={0.2}
              roughness={0.6}
            />
          </mesh>
        );
      }
    });
    
    return { bones, joints };
  }, [keypoints, color, visible, opacity]);
  
  if (!visible) return null;
  
  return (
    <group ref={meshRef}>
      {bones}
      {joints}
    </group>
  );
}

// Animation controls component
function AnimationControls({ 
  isPlaying, 
  onPlayPause, 
  currentFrame, 
  totalFrames, 
  onFrameChange,
  showUser,
  showPro,
  onToggleUser,
  onTogglePro 
}) {
  return (
    <div className="animation-controls">
      <div className="playback-controls">
        <button 
          className={`play-pause-btn ${isPlaying ? 'playing' : 'paused'}`}
          onClick={onPlayPause}
        >
          {isPlaying ? '⏸️' : '▶️'}
        </button>
        
        <div className="timeline-container">
          <input
            type="range"
            min="0"
            max={totalFrames - 1}
            value={currentFrame}
            onChange={(e) => onFrameChange(parseInt(e.target.value))}
            className="timeline-slider"
          />
          <div className="frame-counter">
            {currentFrame + 1} / {totalFrames}
          </div>
        </div>
      </div>
      
      <div className="visibility-controls">
        <label className="toggle-control">
          <input
            type="checkbox"
            checked={showUser}
            onChange={onToggleUser}
          />
          <span className="user-indicator">User Skeleton</span>
        </label>
        
        <label className="toggle-control">
          <input
            type="checkbox"
            checked={showPro}
            onChange={onTogglePro}
          />
          <span className="pro-indicator">Pro Skeleton</span>
        </label>
      </div>
    </div>
  );
}

// Main 3D scene component
function Scene({ userKeypoints, proKeypoints, currentFrame, showUser, showPro, angleAdjustment }) {
  const userFrame = userKeypoints && userKeypoints[currentFrame] ? userKeypoints[currentFrame] : null;
  const proFrame = proKeypoints && proKeypoints[currentFrame] ? proKeypoints[currentFrame] : null;
  
  // Apply pose alignment matching backend logic
  const { userFrame: alignedUserFrame, proFrame: alignedProFrame } = useMemo(() => {
    return alignPoses(userFrame, proFrame, angleAdjustment);
  }, [userFrame, proFrame, angleAdjustment]);
  
  return (
    <>
      {/* Enhanced lighting setup */}
      <ambientLight intensity={0.3} />
      <directionalLight position={[10, 10, 5]} intensity={0.6} castShadow />
      <directionalLight position={[-10, 5, -5]} intensity={0.3} />
      <pointLight position={[0, 5, 0]} intensity={0.4} />
      
      {/* Enhanced ground plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[6, 6]} />
        <meshStandardMaterial 
          color="#f5f5f5" 
          transparent 
          opacity={0.4}
          metalness={0.0}
          roughness={0.8}
        />
      </mesh>
      
      {/* Coordinate system helper */}
      <axesHelper args={[0.5]} />
      
      {/* Skeletons */}
      <Skeleton 
        keypoints={alignedUserFrame} 
        color="#ff4444" 
        visible={showUser}
        opacity={0.8}
      />
      <Skeleton 
        keypoints={alignedProFrame} 
        color="#888888" 
        visible={showPro}
        opacity={0.6}
      />
      
      {/* Labels */}
      {showUser && (
        <Text
          position={[1.5, 1.5, 0]}
          fontSize={0.1}
          color="#ff4444"
          anchorX="center"
          anchorY="middle"
        >
          User
        </Text>
      )}
      {showPro && (
        <Text
          position={[-1.5, 1.5, 0]}
          fontSize={0.1}
          color="#888888"
          anchorX="center"
          anchorY="middle"
        >
          Professional
        </Text>
      )}
      
      {/* Camera controls - enhanced turntable-style rotation */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        target={[0, 0.5, 0]} // Center on skeleton hip height
        minDistance={1.5}
        maxDistance={8}
        minPolarAngle={Math.PI * 0.1} // Prevent camera from going too low
        maxPolarAngle={Math.PI * 0.9} // Prevent camera from going too high
        enableDamping={true}
        dampingFactor={0.05}
        rotateSpeed={0.8}
        zoomSpeed={0.6}
        panSpeed={0.8}
        autoRotate={false}
        autoRotateSpeed={2.0}
      />
    </>
  );
}

// Main MotionViewer3D component
function MotionViewer3D({ taskId, onBack }) {
  const [userKeypoints, setUserKeypoints] = useState(null);
  const [proKeypoints, setProKeypoints] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showUser, setShowUser] = useState(true);
  const [showPro, setShowPro] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [angleAdjustment, setAngleAdjustment] = useState(0);
  
  const animationRef = useRef();
  
  // Load keypoints data
  useEffect(() => {
    const loadKeypoints = async () => {
      try {
        setLoading(true);
        
        // Fetch real 3D keypoints data from API
        const [userKeypointsData, proKeypointsData] = await Promise.all([
          VideoAPI.getUserKeypoints(taskId),
          VideoAPI.getProKeypoints(taskId)
        ]);
        
        setUserKeypoints(userKeypointsData);
        setProKeypoints(proKeypointsData);
        
        // Calculate angle adjustment from first frame (matching backend logic)
        if (userKeypointsData && proKeypointsData && userKeypointsData[0] && proKeypointsData[0]) {
          const userAngle = getStanceAngle(userKeypointsData[0], 'hips');
          const proAngle = getStanceAngle(proKeypointsData[0], 'hips');
          setAngleAdjustment(userAngle - proAngle);
        }
        
        setError(null);
      } catch (err) {
        console.error('Error loading keypoints:', err);
        
        // Fallback to mock data for development/testing
        console.warn('Falling back to mock data due to API error');
        const mockUserKeypoints = generateMockKeypoints(100);
        const mockProKeypoints = generateMockKeypoints(100);
        
        setUserKeypoints(mockUserKeypoints);
        setProKeypoints(mockProKeypoints);
        setError('Using mock data - API endpoints not yet implemented');
      } finally {
        setLoading(false);
      }
    };
    
    if (taskId) {
      loadKeypoints();
    }
  }, [taskId]);
  
  // Animation loop
  useEffect(() => {
    if (isPlaying && userKeypoints) {
      const animate = () => {
        setCurrentFrame(prev => {
          const next = prev + 1;
          if (next >= userKeypoints.length) {
            setIsPlaying(false);
            return prev;
          }
          return next;
        });
      };
      
      animationRef.current = setInterval(animate, 100); // 10 FPS
      
      return () => {
        if (animationRef.current) {
          clearInterval(animationRef.current);
        }
      };
    }
  }, [isPlaying, userKeypoints]);
  
  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };
  
  const handleFrameChange = (frame) => {
    setCurrentFrame(frame);
    setIsPlaying(false);
  };
  
  const totalFrames = userKeypoints ? userKeypoints.length : 0;
  
  if (loading) {
    return (
      <div className="motion-viewer-container">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading 3D motion data...</p>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="motion-viewer-container">
        <div className="error-container">
          <h3>Error Loading 3D Viewer</h3>
          <p>{error}</p>
          <button onClick={onBack} className="back-btn">
            ← Back to Results
          </button>
        </div>
      </div>
    );
  }
  
  return (
    <div className="motion-viewer-container">
      <div className="viewer-header">
        <button onClick={onBack} className="back-btn">
          ← Back to Results
        </button>
        <h2>3D Motion Analysis</h2>
      </div>
      
      <div className="canvas-container">
        <Canvas camera={{ position: [2, 2, 3], fov: 50 }}>
          <Scene 
            userKeypoints={userKeypoints}
            proKeypoints={proKeypoints}
            currentFrame={currentFrame}
            showUser={showUser}
            showPro={showPro}
            angleAdjustment={angleAdjustment}
          />
        </Canvas>
      </div>
      
      <AnimationControls
        isPlaying={isPlaying}
        onPlayPause={handlePlayPause}
        currentFrame={currentFrame}
        totalFrames={totalFrames}
        onFrameChange={handleFrameChange}
        showUser={showUser}
        showPro={showPro}
        onToggleUser={() => setShowUser(!showUser)}
        onTogglePro={() => setShowPro(!showPro)}
      />
    </div>
  );
}

// Helper functions for pose alignment matching backend inference.py

/**
 * Get the stance angle between specific body parts (matching backend get_stance_angle)
 * @param {Array} keypoints - Single frame keypoints [17, 3]
 * @param {string} bodyPart - 'hips', 'shoulders', or 'feet'
 * @returns {number} Angle in degrees
 */
function getStanceAngle(keypoints, bodyPart = 'hips') {
  let left, right;
  
  if (bodyPart === 'hips') {
    left = keypoints[1].slice(0, 2);   // Right Hip (x, y only)
    right = keypoints[4].slice(0, 2);  // Left Hip (x, y only)
  } else if (bodyPart === 'shoulders') {
    left = keypoints[14].slice(0, 2);  // Right Shoulder
    right = keypoints[11].slice(0, 2); // Left Shoulder
  } else { // feet
    left = keypoints[6].slice(0, 2);   // Left Ankle
    right = keypoints[3].slice(0, 2);  // Right Ankle
  }
  
  return findAngle(right, left);
}

/**
 * Find angle between two 2D coordinates (matching backend find_angle)
 * @param {Array} coord1 - [x, y] coordinates
 * @param {Array} coord2 - [x, y] coordinates
 * @returns {number} Angle in degrees
 */
function findAngle(coord1, coord2) {
  const [x1, y1] = coord1;
  const [x2, y2] = coord2;
  
  const dx = x2 - x1;
  const dy = y2 - y1;
  
  if (dx === 0 && dy === 0) {
    return 0; // Points are the same
  }
  
  let angle = Math.atan2(dy, dx) * (180 / Math.PI);
  if (angle < 0) {
    angle += 360;
  }
  return angle;
}

/**
 * Rotate keypoints around Z-axis (matching backend rotate_along_z)
 * @param {Array} keypoints - Single frame keypoints [17, 3]
 * @param {number} degrees - Rotation angle in degrees
 * @returns {Array} Rotated keypoints
 */
function rotateAlongZ(keypoints, degrees) {
  const radians = degrees * (Math.PI / 180);
  const cosTheta = Math.cos(radians);
  const sinTheta = Math.sin(radians);
  
  return keypoints.map(([x, y, z]) => [
    x * cosTheta - y * sinTheta,
    x * sinTheta + y * cosTheta,
    z
  ]);
}

/**
 * Recenter pose on specific joint (matching backend recenter_on_joint)
 * @param {Array} keypoints - Single frame keypoints [17, 3]
 * @param {number} jointIdx - Index of joint to center on (0 = hip center)
 * @param {Array} target - Target position [x, y, z]
 * @returns {Array} Recentered keypoints
 */
function recenterOnJoint(keypoints, jointIdx = 0, target = [0, 0, 0]) {
  const offset = keypoints[jointIdx];
  return keypoints.map(([x, y, z]) => [
    x - offset[0] + target[0],
    y - offset[1] + target[1],
    z - offset[2] + target[2]
  ]);
}

/**
 * Process pose alignment for overlay (matching backend create_pose_overlay_image logic)
 * @param {Array} userFrame - User keypoints for current frame [17, 3]
 * @param {Array} proFrame - Pro keypoints for current frame [17, 3]
 * @param {number} angleAdjustment - Angle adjustment from first frame alignment
 * @returns {Object} Aligned poses
 */
function alignPoses(userFrame, proFrame, angleAdjustment = 0) {
  if (!userFrame || !proFrame) return { userFrame, proFrame };
  
  // Rotate pro pose to align with user pose
  let alignedProFrame = rotateAlongZ(proFrame, angleAdjustment);
  
  // Recenter both poses on their hip centers (joint 0)
  const centeredUserFrame = recenterOnJoint(userFrame, 0, [0, 0, 0]);
  const centeredProFrame = recenterOnJoint(alignedProFrame, 0, [0, 0, 0]);
  
  return {
    userFrame: centeredUserFrame,
    proFrame: centeredProFrame
  };
}

// Temporary function to generate mock keypoints data
// This will be replaced with actual API calls
function generateMockKeypoints(numFrames) {
  const keypoints = [];
  
  for (let frame = 0; frame < numFrames; frame++) {
    const frameKeypoints = [];
    
    // Generate 17 joints with some animation
    for (let joint = 0; joint < 17; joint++) {
      const time = frame * 0.1;
      const x = Math.sin(time + joint * 0.2) * 0.2;
      const y = joint < 7 ? joint * 0.15 : (joint - 7) * 0.1 + 1.0; // legs lower, upper body higher
      const z = Math.cos(time + joint * 0.1) * 0.1;
      
      frameKeypoints.push([x, y, z]);
    }
    
    keypoints.push(frameKeypoints);
  }
  
  return keypoints;
}

export default MotionViewer3D;