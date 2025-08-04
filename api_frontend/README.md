# Shadow Trainer Frontend

React-based frontend application providing an intuitive user interface for video upload, processing visualization, and 3D motion analysis results for the Shadow Trainer platform.

## 🏗️ Frontend Architecture

The frontend communicates with the backend API hosted at [api.shadow-trainer.com](https://api.shadow-trainer.com).
For detailed API documentation and usage examples, visit the Shadow Trainer API Swagger docs reference at [api.shadow-trainer.com/docs](https://api.shadow-trainer.com/docs).

### Core Stack
- **React 18.2+** - Component-based UI framework with hooks
- **React Router DOM 6.30+** - Client-side routing and navigation
- **Three.js 0.178+** - 3D graphics and WebGL rendering
- **@react-three/fiber** - React renderer for Three.js
- **@react-three/drei** - Useful helpers for react-three-fiber
- **Axios 1.6+** - HTTP client for API communication

### UI Framework
- **CSS Modules** - Scoped component styling
- **Responsive Design** - Mobile-first approach with breakpoints
- **Component Architecture** - Reusable UI components and pages

## 📁 Project Structure

```
api_frontend/
├── public/                    # Static assets
├── src/                      # Source code
│   ├── App.js                # Main application component
│   ├── App.css               # Global application styles
│   ├── components/           # Reusable UI components
│   │   ├── pages/            # Page-level components
│   │   └── ui/               # Reusable UI components
│   ├── services/             # API communication layer
│   └── styles/               # Component-specific CSS modules
├── package.json              # Dependencies and scripts
├── Dockerfile               # Production container build
└── build/                   # Production build output (generated)
```

## 🎯 Key Features

### Video Upload & Processing
- **Drag & Drop Interface** - Intuitive file upload with validation
- **Format Support** - MP4, MOV, AVI video formats
- **Progress Tracking** - Real-time processing status updates
- **Error Handling** - User-friendly error messages and retry options

### 3D Visualization
- **WebGL Rendering** - Hardware-accelerated 3D graphics with Three.js
- **Dual Skeleton Display** - Side-by-side user vs professional comparison
- **Interactive Controls** - Camera rotation, zoom, and playback controls
- **Frame-by-Frame Analysis** - Scrub through motion frame by frame
- **Multiple View Angles** - Front, side, and custom camera perspectives

### Analysis Dashboard
- **Biomechanical Charts** - Joint angle and velocity visualizations
- **Comparison Metrics** - Statistical analysis vs professional athletes
- **AI Coaching Feedback** - Markdown-formatted improvement suggestions
- **Export Capabilities** - Download analysis charts and data

### User Experience
- **Responsive Design** - Works on desktop, tablet, and mobile devices
- **Loading States** - Smooth transitions and progress indicators
- **Professional UI** - Clean, modern interface design
- **Accessibility** - Keyboard navigation and screen reader support

## 🔗 API Integration

The frontend communicates with the FastAPI backend through the `videoApi.js` service layer:

```javascript
// Core API functions
uploadAndProcessVideo()     // Upload video and start processing
getProcessingStatus()       // Check task progress
downloadProcessedVideo()    // Get final results
getUserKeypoints()          // Fetch 3D user keypoints
getProKeypoints()          // Fetch professional athlete data
getAnalysisPlots()         // Retrieve analysis charts
generateEvaluation()       // Request AI coaching feedback
```

## 🧩 Component Architecture

### Page Components
- **LandingPage** - Entry point with video upload and instructions
- **ProcessingPage** - Processing status with progress bar and logs
- **ResultsPage** - Analysis results with charts and AI feedback
- **VisualizerPage** - 3D skeleton visualization and playback controls
- **AppPage** - Main application router and state management

### UI Components
- **FileUpload** - Handles video file selection and validation
- **Skeletonviewer** - 3D rendering of motion capture keypoints
- **PlotViewer** - Display analysis charts and biomechanical data
- **ProgressBar** - Visual progress indication during processing
- **VideoPreview** - Video playback with custom controls

## 🎨 Styling Architecture

### CSS Organization
- **Global Styles** (`globals.css`) - CSS custom properties and resets
- **Component Styles** - Individual CSS files for each component
- **Modular Approach** - Scoped styles to prevent style conflicts
- **Responsive Design** - Mobile-first with CSS Grid and Flexbox

### Design System
```css
/* CSS Custom Properties (globals.css) */
:root {
  --primary-color: #007bff;
  --secondary-color: #6c757d;
  --success-color: #28a745;
  --danger-color: #dc3545;
  --background-color: #f8f9fa;
  --text-color: #212529;
}
```

## 🛠️ Development

### Local Setup
```bash
cd api_frontend
npm install                 # Install dependencies
npm start                  # Start development server (port 3000)
npm run build              # Create production build
npm test                   # Run test suite
```

### Development Server
- **Hot Reloading** - Automatic browser refresh on code changes
- **Proxy Configuration** - Development API calls proxied to backend
- **Source Maps** - Debug support with original source code mapping
- **ESLint Integration** - Code quality and style enforcement

### Adding New Components

**Page Components:**
1. Create new component in `src/components/pages/`
2. Add corresponding CSS file in `src/styles/`
3. Update router configuration in `App.js`
4. Add navigation links where appropriate

**UI Components:**
1. Create reusable component in `src/components/ui/`
2. Add scoped styles in `src/styles/`
3. Export component for use in pages
4. Document props and usage examples

### State Management
- **React Hooks** - useState and useEffect for component state
- **Props Drilling** - Data passed down through component hierarchy
- **Context API** - For shared state across multiple components
- **API State** - Managed through service layer and component state

## 🚀 Production Build

### Docker Deployment
```bash
# Build production container
docker build -t shadow-trainer-frontend .

# Run container (port 3000)
docker run -p 3000:3000 shadow-trainer-frontend
```

### Build Optimization
- **Code Splitting** - Automatic bundle splitting for optimal loading
- **Tree Shaking** - Remove unused code from final bundle
- **Asset Optimization** - Image compression and static asset optimization
- **Caching Strategy** - Long-term caching for static assets

### Environment Configuration
```javascript
// Environment variables (available in React)
REACT_APP_API_URL=http://localhost:8002  // Backend API endpoint
REACT_APP_VERSION=1.0.0                  // Application version
```

## 🧪 Testing

```bash
npm test                    # Run test suite
npm test -- --coverage     # Generate coverage report
npm test -- --watchAll     # Run tests in watch mode
```

## 📱 Browser Support

- **Chrome 90+** - Full feature support
- **Firefox 88+** - Full feature support  
- **Safari 14+** - Full feature support
- **Edge 90+** - Full feature support
- **Mobile Browsers** - Responsive design with touch support

## 🔧 Troubleshooting

**Common Issues:**
- **CORS Errors** - Ensure backend allows frontend origin
- **3D Rendering Issues** - Check WebGL support and GPU drivers
- **Upload Failures** - Verify file size limits and network connectivity
- **Slow Performance** - Check for memory leaks in 3D components