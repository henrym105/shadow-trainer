import React from 'react';
import axios from 'axios';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Custom error class for API errors
export class APIError extends Error {
  constructor(message, status, response) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.response = response;
  }
}

// API service class
export class VideoAPI {
  
  /**
   * Upload video file and start processing
   * @param {File} file - Video file to upload
   * @param {string} modelSize - Model size ('xs', 's', 'm', 'l')
   * @param {boolean} isLefty - Whether user is left-handed
   * @param {string} proKeypointsFilename - Professional player keypoints filename
   * @param {string} videoFormat - Output video format ('combined' or '3d_only')
   * @returns {Promise<Object>} Task information
   */
  static async uploadVideo(file, modelSize = 'xs', isLefty = false, proKeypointsFilename = 'BlakeSnell_median.npy', videoFormat = 'combined') {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const params = new URLSearchParams({
        model_size: modelSize,
        is_lefty: isLefty.toString(),
        pro_keypoints_filename: proKeypointsFilename,
        video_format: videoFormat
      });

      const response = await api.post(`/videos/upload?${params}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Upload error:', error);
      throw new APIError(
        error.response?.data?.detail || 'Failed to upload video',
        error.response?.status,
        error.response
      );
    }
  }

  /**
   * Process sample lefty video (we'll need to add this endpoint to the backend)
   * @param {string} modelSize - Model size ('xs', 's', 'm', 'l')
   * @param {string} proKeypointsFilename - Professional player keypoints filename
   * @param {string} videoFormat - Output video format ('combined' or '3d_only')
   * @returns {Promise<Object>} Task information
   */
  static async processSampleLeftyVideo(modelSize = 'xs', proKeypointsFilename = 'BlakeSnell_median.npy', videoFormat = 'combined') {
    try {
      const params = new URLSearchParams({
        model_size: modelSize,
        is_lefty: 'true',
        pro_keypoints_filename: proKeypointsFilename,
        video_format: videoFormat
      });

      const response = await api.post(`/videos/sample-lefty?${params}`);
      return response.data;
    } catch (error) {
      console.error('Sample processing error:', error);
      throw new APIError(
        error.response?.data?.detail || 'Failed to process sample video',
        error.response?.status,
        error.response
      );
    }
  }

  /**
   * Get job status
   * @param {string} taskId - Task identifier
   * @returns {Promise<Object>} Job status information
   */
  static async getJobStatus(taskId) {
    try {
      const response = await api.get(`/videos/${taskId}/status`);
      return response.data;
    } catch (error) {
      console.error('Status check error:', error);
      throw new APIError(
        error.response?.data?.detail || 'Failed to get job status',
        error.response?.status,
        error.response
      );
    }
  }

  /**
   * Get download URL for processed video
   * @param {string} taskId - Task identifier
   * @returns {string} Download URL
   */
  static getDownloadUrl(taskId) {
    return `${API_BASE_URL}/videos/${taskId}/download`;
  }

  /**
   * Get preview URL for processed video
   * @param {string} taskId - Task identifier
   * @returns {string} Preview URL
   */
  static getPreviewUrl(taskId) {
    return `${API_BASE_URL}/videos/${taskId}/preview`;
  }

  /**
   * Validate file client-side
   * @param {File} file - File to validate
   * @returns {Object} Validation result
   */
  static validateFile(file) {
    const validTypes = ['.mp4', '.mov', '.avi', '.mkv'];
    const maxSize = 100 * 1024 * 1024; // 100MB
    
    if (!file) {
      return { valid: false, error: 'No file selected' };
    }

    const extension = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
    if (!validTypes.includes(extension)) {
      return { 
        valid: false, 
        error: `Invalid file type. Please upload a video file (${validTypes.join(', ')})` 
      };
    }

    if (file.size > maxSize) {
      return { 
        valid: false, 
        error: 'File size too large. Please upload a video smaller than 100MB' 
      };
    }

    return { valid: true };
  }

  /**
   * Health check
   * @returns {Promise<boolean>} API availability
   */
  static async healthCheck() {
    try {
      const response = await api.get('/');
      return response.status === 200;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  /**
   * Get list of professional keypoints files
   * @returns {Promise<Array>} List of available professional players
   */
  static async getProKeypointsList() {
    try {
      const response = await api.get('/pro_keypoints/list');
      return response.data.files || [];
    } catch (error) {
      console.error('Failed to get pro keypoints list:', error);
      throw new APIError(
        error.response?.data?.detail || 'Failed to get professional players list',
        error.response?.status,
        error.response
      );
    }
  }

  /**
   * Terminate a running video processing task
   * @param {string} taskId - Task identifier
   * @returns {Promise<Object>} Termination result
   */
  static async terminateTask(taskId) {
    try {
      const response = await api.post(`/videos/${taskId}/terminate`);
      return response.data;
    } catch (error) {
      console.error('Failed to terminate task:', error);
      throw new APIError(
        error.response?.data?.detail || 'Failed to terminate task',
        error.response?.status,
        error.response
      );
    }
  }

  /**
   * Upload video file and start 3D keypoints extraction
   * @param {File} file - Video file to upload
   * @param {string} modelSize - Model size ('xs', 's', 'm', 'l')
   * @returns {Promise<Object>} Task information
   */
  static async upload3DKeypoints(file, modelSize = 'xs') {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const params = new URLSearchParams({
        model_size: modelSize
      });

      const response = await api.post(`/videos/get_3d_keypoints?${params}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('3D keypoints upload error:', error);
      throw new APIError(
        error.response?.data?.detail || 'Failed to upload video for 3D keypoints extraction',
        error.response?.status,
        error.response
      );
    }
  }

  /**
   * Get download URL for 3D keypoints .npy file
   * @param {string} taskId - Task identifier
   * @returns {string} Download URL
   */
  static getKeypointsDownloadUrl(taskId) {
    return `${API_BASE_URL}/files/${taskId}/download`;
  }
}

// Custom hook for job polling
export const useJobPolling = (taskId, onStatusUpdate, pollingInterval = 2000) => {
  const [isPolling, setIsPolling] = React.useState(false);

  React.useEffect(() => {
    if (!taskId) return;

    setIsPolling(true);
    const pollStatus = async () => {
      try {
        const status = await VideoAPI.getJobStatus(taskId);
        onStatusUpdate(status);

        // Stop polling if job is completed, failed, or terminated
        if (status.status === 'completed' || status.status === 'failed' || status.status === 'terminated') {
          setIsPolling(false);
          return;
        }

        // Continue polling
        setTimeout(pollStatus, pollingInterval);
      } catch (error) {
        console.error('Polling error:', error);
        onStatusUpdate({ 
          status: 'failed', 
          error: 'Failed to check job status' 
        });
        setIsPolling(false);
      }
    };

    pollStatus();

    return () => setIsPolling(false);
  }, [taskId, onStatusUpdate, pollingInterval]);

  return { isPolling };
};

export default VideoAPI;
