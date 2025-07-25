# Shadow Trainer Frontend Architecture Documentation

## Overview

Shadow Trainer is a React-based web application for AI-powered motion analysis and athletic performance evaluation. The frontend provides an intuitive interface for users to upload training videos, configure analysis parameters, monitor processing progress, and view results. The application interfaces with a Celery-based backend API for video processing tasks.

## Technology Stack

- **Framework**: React 18.2.0
- **HTTP Client**: Axios 1.6.0
- **Build Tool**: Create React App (react-scripts 5.0.1)
- **Styling**: Pure CSS with custom component stylesheets
- **Deployment**: Docker container with nginx serve

## Application Architecture

### Core Application Structure

```
src/
├── App.js                     # Main application component
├── App.css                    # Global application styles
├── index.js                   # React application entry point
├── sampleVideos.js           # Static sample video configuration
├── components/               # Reusable UI components
│   ├── FileUpload.js/css    # Drag-and-drop file upload component
│   ├── ProgressBar.js/css   # Processing progress indicator
│   ├── VideoResult.js/css   # Results display and download
│   └── ProKeypointsSelector.js # Professional player comparison selector
└── services/
    ├── videoApi.js          # API service layer and hooks
    └── videoApi_new.js      # Alternative API implementation
```

### Application State Management

The application uses React's built-in `useState` hooks for state management with the following key state variables:

- `selectedFile`: Currently selected video file object
- `uploadError`: Error messages for user feedback
- `taskId`: Celery task identifier for processing jobs
- `jobStatus`: Current processing status and progress
- `isUploading`: Upload operation state
- `modelSize`: AI model quality selection ('xs', 's', 'm', 'l')
- `isLefty`: User handedness preference (boolean)
- `selectedProFile`: Professional player keypoints file selection

### Application Flow States

The application operates in four distinct states:

1. **Upload State**: Initial file selection and configuration
2. **Processing State**: Real-time progress monitoring
3. **Completed State**: Results display and download options
4. **Failed State**: Error handling and retry options

## Component Details

### App.js - Main Application Component

**Responsibilities**:
- Central state management
- Application flow orchestration
- API integration
- Error handling and user feedback

**Key Features**:
- File upload with validation
- Sample video processing
- Real-time job status polling
- Responsive state transitions
- Professional player comparison selection

**API Integration**:
- Video upload: `POST /videos/upload`
- Sample processing: `POST /videos/sample-lefty`
- Status polling: `GET /videos/{task_id}/status`
- Result streaming: `GET /videos/{task_id}/preview`
- File download: `GET /videos/{task_id}/download`

### FileUpload.js - File Selection Component

**Responsibilities**:
- Drag-and-drop file interface
- File validation (type, size)
- Video preview generation
- User feedback for invalid files

**Features**:
- Supports multiple video formats (.mp4, .mov, .avi, .mkv)
- 100MB file size limit
- Visual drag-and-drop feedback
- Immediate video preview after selection
- Clear selection functionality

**Validation Rules**:
- File type checking by extension and MIME type
- Maximum file size enforcement
- User-friendly error messages

### ProgressBar.js - Processing Status Component

**Responsibilities**:
- Visual progress indication
- Status message display
- Processing step visualization
- Animated progress feedback

**Features**:
- Percentage-based progress tracking
- Status-specific styling and icons
- Smooth animation effects
- Processing step breakdown display

**Status Mapping**:
- `queued`: Orange with clock icon
- `processing`: Green with gear icon
- `completed`: Blue with checkmark icon
- `failed`: Red with error icon

### VideoResult.js - Results Display Component

**Responsibilities**:
- Processed video preview
- Download functionality
- Link sharing capabilities
- Result metadata display

**Features**:
- Embedded video player with controls
- Direct download button
- Clipboard link copying
- Job ID and processing info display
- Error handling for video loading failures

### ProKeypointsSelector.js - Professional Comparison Component

**Responsibilities**:
- Professional player selection
- Dynamic option loading from API
- User preference persistence

**Features**:
- Fetches available professional players from `/pro_keypoints/list`
- Dropdown selection interface
- Player name and team information display
- Integration with processing parameters

## API Service Layer

### videoApi.js - Core API Service

**Main Service Class: VideoAPI**

**Methods**:
- `uploadVideo(file, modelSize, isLefty, proKeypointsFilename)`: File upload and processing initiation
- `processSampleLeftyVideo(modelSize, proKeypointsFilename)`: Sample video processing
- `getJobStatus(taskId)`: Polling for job status updates
- `getDownloadUrl(taskId)`: Generate download URLs
- `getPreviewUrl(taskId)`: Generate preview stream URLs
- `validateFile(file)`: Client-side file validation
- `healthCheck()`: API availability verification

**Custom Hook: useJobPolling**

**Purpose**: Automated status polling for long-running tasks

**Features**:
- Configurable polling interval (default: 2 seconds)
- Automatic polling termination on completion/failure
- Error handling and retry logic
- React lifecycle integration

**Parameters**:
- `taskId`: Job identifier to monitor
- `onStatusUpdate`: Callback for status changes
- `pollingInterval`: Update frequency in milliseconds

## Styling Architecture

### Design System

**Color Palette**:
- Primary: Green gradient (#366e2f to #2e4c2b)
- Secondary: White with transparency
- Success: #4CAF50
- Warning: #ffa500
- Error: #f44336
- Info: #2196F3

**Typography**:
- Font Family: Apple system fonts stack
- Heading sizes: 2.5rem (h1) to 1.1rem (p)
- Font weights: 700 (bold), 600 (semi-bold), 400 (normal)

**Layout Principles**:
- Responsive design with max-width containers
- Flexbox-based layouts
- Mobile-first approach
- Consistent spacing using rem units

### Component-Specific Styling

**FileUpload.css**:
- Drag-and-drop visual feedback
- State-based styling (normal, hover, active, disabled)
- File information display
- Video preview integration

**ProgressBar.css**:
- Animated progress indicators
- Status-specific color coding
- Shimmer effects for active processing
- Responsive text sizing

**VideoResult.css**:
- Video player styling
- Button design system
- Loading and error states
- Information card layouts

## Configuration and Environment

### Environment Variables

- `REACT_APP_API_URL`: Backend API base URL
  - Development: `http://localhost:8000`
  - Production: `https://api.shadow-trainer.com`

### Build Configuration

**package.json dependencies**:
- React 18.2.0 for modern React features
- Axios 1.6.0 for HTTP requests
- React-scripts 5.0.1 for build tooling

**Build process**:
- Development: `npm start` (hot reloading)
- Production: `npm run build` (optimized bundle)
- Testing: `npm test` (Jest test runner)

### Deployment Architecture

**Docker Configuration**:
- Multi-stage build for optimization
- Node.js 18 Alpine base image
- Static file serving with `serve` package
- Port 3000 exposure

**Build Process**:
1. Install dependencies
2. Create optimized production build
3. Serve static files with nginx-like server

## User Experience Flow

### Upload Flow

1. **Initial State**: User sees upload interface with drag-and-drop zone
2. **File Selection**: User drags file or clicks to browse
3. **Validation**: Real-time file validation with immediate feedback
4. **Configuration**: Model size, handedness, and professional player selection
5. **Submission**: Upload initiation with loading feedback

### Processing Flow

1. **Status Polling**: Automatic background polling every 2 seconds
2. **Progress Visualization**: Step-by-step progress with percentage tracking
3. **Step Breakdown**: Visual indicators for processing stages
4. **Real-time Updates**: Dynamic status messages and progress updates

### Results Flow

1. **Completion Detection**: Automatic transition to results view
2. **Video Preview**: Embedded player with processed video
3. **Download Options**: Direct download and link sharing
4. **Metadata Display**: Job information and processing details

## Error Handling Strategy

### Client-Side Validation

- File type and size validation before upload
- User-friendly error messages
- Visual feedback for invalid states
- Clear resolution guidance

### API Error Handling

- Custom `APIError` class for structured error information
- HTTP status code mapping
- Retry mechanisms for transient failures
- Graceful degradation for service unavailability

### Network Error Recovery

- Automatic retry for failed requests
- Offline state detection and user notification
- Fallback content for critical features
- Loading state management

## Performance Considerations

### File Handling

- Client-side file validation to reduce server load
- Progressive upload with feedback
- Video preview generation using object URLs
- Memory management for large files

### Polling Optimization

- Efficient polling with automatic termination
- Exponential backoff for failed requests
- Debounced status updates
- Background polling without UI blocking

### Bundle Optimization

- Code splitting for reduced initial load
- Lazy loading for non-critical components
- Asset optimization through Create React App
- Production build minification

## Security Considerations

### File Upload Security

- Client-side file type validation
- Size limit enforcement
- Secure file URL generation
- Temporary file cleanup

### API Communication

- CORS configuration for cross-origin requests
- Environment-based URL configuration
- Error message sanitization
- No sensitive data in client-side code

## Extension Points (future work)

### Additional Features

1. **User Authentication**: Login/logout functionality
2. **Project Management**: Save and organize multiple videos
3. **Advanced Analytics**: Detailed motion analysis metrics
4. **Social Features**: Sharing and collaboration tools
5. **Batch Processing**: Multiple video upload and processing

### Component Extensibility

1. **Custom Players**: Alternative video player implementations
2. **Additional Formats**: Support for more file types
3. **Cloud Integration**: Direct cloud storage access
4. **Real-time Streaming**: Live video analysis
5. **Mobile Optimization**: Touch-friendly interfaces

### API Integration

1. **Webhook Support**: Real-time status updates
2. **Streaming Upload**: Large file handling
3. **Background Processing**: Queue management
4. **Result Caching**: Performance optimization
5. **Analytics Integration**: Usage tracking and metrics

## Development Guidelines

### Code Organization

- Component-based architecture with single responsibility
- Separation of concerns between UI and business logic
- Consistent naming conventions and file structure
- Comprehensive error handling and user feedback

This architecture provides a solid foundation for a professional video processing application while maintaining flexibility for future enhancements and optimizations.
