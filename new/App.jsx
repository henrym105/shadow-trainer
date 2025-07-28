import SkeletonViewer from './Skeletonviewer';
// import userKeypoints from './keypoints/userKeypoints.json';
// import proKeypoints from './keypoints/proKeypoints.json';

// import userKeypoints from './keypoints/pro_3D_keypoints.json';
// import proKeypoints from './keypoints/user_3D_keypoints.json';

// import userKeypoints from './keypoints/pro_3D_keypoints.json';
// import proKeypoints from './keypoints/user_3D_keypoints.json';

import userKeypoints from './keypoints/pro_3D_keypoints_Lopez.json';
import proKeypoints from './keypoints/user_3D_keypoints_Henry.json';

function App() {
  return (
    <div style={{ height: '100vh' }}>
      <SkeletonViewer
        keypointFrames={userKeypoints}
        proKeypointFrames={proKeypoints}
      />
    </div>
  );
}

export default App;