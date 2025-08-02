import { useRef, useState, useEffect } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';

function Skeleton({ frames, color = 'red', frame, isLefty = false }) {
  // Always use the frame prop to select the current frame
  const currentFrame = frames[frame] || frames[0];

  const root = [
    currentFrame[0],     // x of keypoint 0
    currentFrame[1],     // y
    currentFrame[2]      // z
  ];

  const points = [];
  for (let i = 0; i < currentFrame.length; i += 3) {
    points.push([
      currentFrame[i] - root[0],
      currentFrame[i + 1] - root[1],
      currentFrame[i + 2] - root[2]
    ]);
  }

  const bones = [
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
    [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
    [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]
  ];

  // Define throwing hand wrist joint for highlighting
  // Joint numbers: 15=LeftWrist, 16=RightWrist
  const throwingHandWrist = isLefty ? 15 : 16;  // Left wrist for lefties, right wrist for righties

  return (
    <>
      {points.map(([x, y, z], i) => {
        // Check if this is the throwing hand wrist
        const isThrowingWrist = i === throwingHandWrist;
        const jointColor = isThrowingWrist ? color : color;
        
        return (
          <mesh key={i} position={[x, y, z]}>
            {/* <sphereGeometry args={[sphereSize, 8, 8]} /> */}
            <sphereGeometry args={[0.02, 8, 8]} />
            <meshStandardMaterial
              color={jointColor}
              opacity={0.6}
              transparent={true}
            />
          </mesh>
        );
      })}

      {bones.map(([start, end], i) => {
        if (!points[start] || !points[end]) return null;
        const linePoints = [
          points[start],
          points[end]
        ];
        return (
          <Line
            key={`bone-${i}`}
            points={linePoints}
            color={color}
            lineWidth={10}
            opacity={0.5}
            transparent={true}
          />
        );
      })}
    </>
  );
}

function FixedZOrbitControls() {
  const { camera, gl } = useThree();
  camera.up.set(0, 0, 1); // Z is up
  return (
    <OrbitControls
      args={[camera, gl.domElement]}
      enableZoom={true}
      enablePan={false}
      minPolarAngle={Math.PI / 2}
      maxPolarAngle={Math.PI}
    />
  );
}

// Add this component for turntable animation
function FloorGrid({ keypointFrames }) {
  // Calculate the lowest z-value from the first frame, accounting for root transformation
  let minZ = 0;
  if (keypointFrames && keypointFrames.length > 0) {
    const firstFrame = keypointFrames[0];
    const root = [firstFrame[0], firstFrame[1], firstFrame[2]];
    
    // Transform all points relative to root and find minimum z
    for (let i = 0; i < firstFrame.length; i += 3) {
      const transformedZ = firstFrame[i + 2] - root[2];
      minZ = Math.min(minZ, transformedZ);
    }
  }
  
  return (
    <mesh position={[0, 0, minZ - 0.1]}>
      <planeGeometry args={[4, 4, 10, 10]} />
      <meshBasicMaterial color="#7b7b7bff" wireframe={true} opacity={0.5} transparent={true} wireframeLinewidth={3} />
    </mesh>
  );
}

function TurntableControls() {
  const { camera } = useThree();
  const radius = useRef(camera.position.length());
  useEffect(() => {
    radius.current = camera.position.length();
  }, [camera.position]);
  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    const r = radius.current;
    // Rotate around Z axis at 10 seconds per revolution
    camera.position.x = r * Math.cos(t * 2 * Math.PI / 10);
    camera.position.y = r * Math.sin(t * 2 * Math.PI / 10);
    camera.position.z = camera.position.z; // keep z unchanged
    camera.lookAt(0, 0, 0);
  });
  return null;
}

export default function SkeletonViewer({ 
  keypointFrames, 
  proKeypointFrames, 
  playing = true,
  showUserSkeleton = true,
  showProSkeleton = true,
  frame = 0,
  turntable = false,
  playbackSpeed = 1,
  onFrameChange,
  isLefty = false
}) {
  const totalFrames = keypointFrames?.length || 1;

  // Animation loop using setInterval with playback speed
  useEffect(() => {
    if (!playing) return;
    const frameRate = 33 / playbackSpeed; // Adjust frame rate based on speed
    const id = setInterval(() => {
      const newFrame = (frame + 1) % totalFrames;
      if (onFrameChange) {
        onFrameChange(newFrame);
      }
    }, frameRate);
    return () => clearInterval(id);
  }, [playing, totalFrames, playbackSpeed, frame, onFrameChange]);

  return (
    <Canvas 
      camera={{ position: [2, 0, 1.2], up: [0, 0, 1] }} 
      style={{ 
        width: '100%', 
        height: '100%',
        maxHeight: 'min(75vh, 650px)'
      }}
    >
      <ambientLight />
      <FloorGrid keypointFrames={keypointFrames} />
      {turntable ? <TurntableControls /> : <FixedZOrbitControls />}
      {showUserSkeleton && (
        <Skeleton frames={keypointFrames} color="red" frame={frame} isLefty={isLefty} />
      )}
      {showProSkeleton && proKeypointFrames && (
        <Skeleton frames={proKeypointFrames} color="black" frame={frame} isLefty={isLefty} />
      )}
    </Canvas>
  );
}