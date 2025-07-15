import React, { useRef, useEffect } from "react";
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

function drawSkeleton(scene, keypoints, color = 0x00ff00) {
  // Remove previous skeleton
  while (scene.children.length > 0) scene.remove(scene.children[0]);
  // Draw joints
  keypoints.forEach(([x, y, z]) => {
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.03, 8, 8),
      new THREE.MeshBasicMaterial({ color })
    );
    sphere.position.set(x, y, z);
    scene.add(sphere);
  });
  // Draw bones
  SKELETON_EDGES.forEach(([i, j]) => {
    if (keypoints[i] && keypoints[j]) {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...keypoints[i]),
        new THREE.Vector3(...keypoints[j])
      ]);
      const line = new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({ color, linewidth: 2 })
      );
      scene.add(line);
    }
  });
}

const ThreeJSkeleton = ({ userKeypoints, proKeypoints, showUser, showPro, frameIdx }) => {
  const mountRef = useRef();
  const sceneRef = useRef();
  const rendererRef = useRef();
  const cameraRef = useRef();

  useEffect(() => {
    // Init scene
    const width = 500;
    const height = 500;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(0, 0, 2.5);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    mountRef.current.appendChild(renderer.domElement);
    sceneRef.current = scene;
    rendererRef.current = renderer;
    cameraRef.current = camera;
    // Light
    const light = new THREE.AmbientLight(0xffffff, 1);
    scene.add(light);
    // Clean up
    return () => {
      renderer.dispose();
      mountRef.current.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    // Draw user skeleton
    if (showUser && userKeypoints && userKeypoints[frameIdx]) {
      drawSkeleton(scene, userKeypoints[frameIdx], 0x00ff00);
    }
    // Draw pro skeleton
    if (showPro && proKeypoints && proKeypoints[frameIdx]) {
      drawSkeleton(scene, proKeypoints[frameIdx], 0xff0000);
    }
    // If both, draw both (user on top)
    if (showUser && showPro && userKeypoints && proKeypoints && userKeypoints[frameIdx] && proKeypoints[frameIdx]) {
      drawSkeleton(scene, proKeypoints[frameIdx], 0xff0000);
      drawSkeleton(scene, userKeypoints[frameIdx], 0x00ff00);
    }
    rendererRef.current.render(scene, cameraRef.current);
  }, [userKeypoints, proKeypoints, showUser, showPro, frameIdx]);

  return <div ref={mountRef} />;
};

export default ThreeJSkeleton;
