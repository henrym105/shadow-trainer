import React, { useRef, useState } from "react";
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from "three";

// Example joint connections for a 17-joint skeleton (COCO order)
const SKELETON_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4], // Right arm
  [0, 5], [5, 6], [6, 7], [7, 8], // Left arm
  [0, 9], [9, 10], [10, 11], // Torso right
  [0, 12], [12, 13], [13, 14], // Torso left
  [9, 12], // Hip connection
  [11, 15], [14, 16] // Legs (example)
];

function SkeletonLines({ keypoints, color }) {
  if (!keypoints) return null;
  return (
    <>
      {/* Joints */}
      {keypoints.map(([x, y, z], idx) => (
        <mesh key={"joint-" + idx} position={[x, y, z]}>
          <sphereGeometry args={[0.03, 8, 8]} />
          <meshBasicMaterial color={color} />
        </mesh>
      ))}
      {/* Bones */}
      {SKELETON_EDGES.map(([i, j], idx) => (
        (keypoints[i] && keypoints[j]) ? (
          <line key={"bone-" + idx}>
            <bufferGeometry attach="geometry">
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([
                  ...keypoints[i],
                  ...keypoints[j]
                ])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color={color} linewidth={2} />
          </line>
        ) : null
      ))}
    </>
  );
}

function TurntableGroup({ children }) {
  const group = useRef();
  const [dragging, setDragging] = useState(false);
  const [lastX, setLastX] = useState(0);
  const [rotationY, setRotationY] = useState(0);

  useFrame(() => {
    if (group.current) {
      group.current.rotation.y = rotationY;
    }
  });

  const onPointerDown = (e) => {
    setDragging(true);
    setLastX(e.clientX);
  };
  const onPointerUp = () => setDragging(false);
  const onPointerMove = (e) => {
    if (!dragging) return;
    const deltaX = e.clientX - lastX;
    setLastX(e.clientX);
    setRotationY((r) => r + deltaX * 0.01);
  };

  return (
    <group
      ref={group}
      onPointerDown={onPointerDown}
      onPointerUp={onPointerUp}
      onPointerOut={onPointerUp}
      onPointerMove={onPointerMove}
    >
      {children}
    </group>
  );
}

const ThreeJSkeleton = ({ userKeypoints, proKeypoints, showUser, showPro, frameIdx }) => {
  // Defensive: avoid rendering if no data
  const userFrame = userKeypoints && userKeypoints[frameIdx];
  const proFrame = proKeypoints && proKeypoints[frameIdx];

  return (
    <Canvas style={{ width: 500, height: 500, background: '#fff', borderRadius: 8 }} camera={{ position: [0, 0, 2.5], fov: 75 }}>
      <ambientLight intensity={1} />
      <TurntableGroup>
        {showPro && proFrame && <SkeletonLines keypoints={proFrame} color={0xff0000} />}
        {showUser && userFrame && <SkeletonLines keypoints={userFrame} color={0x00ff00} />}
      </TurntableGroup>
    </Canvas>
  );
};

export default ThreeJSkeleton;
