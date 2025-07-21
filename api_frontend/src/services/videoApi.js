/**
 * API Service for Shadow Trainer
 * Handles all communication with the backend API
 */

import { useState, useEffect, useCallback } from 'react';

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

/**
 * API Error class for better error handling
 */
export class APIError extends Error {
  constructor(message, status = 500, detail = '') {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Main API service class
 */
export class VideoAPI {
  /**
   * Upload a video file and start processing
   * @param {File} file - The video file to upload
   * @param {string} modelSize - Model size to use ('xs', 's', 'm', 'l')
   * @param {boolean} isLefty - Whether the user is left-handed
   * @param {string} proKeypointsFilename - Optional pro keypoints filename
   * @returns {Promise<Object>} Upload response with task_id
   */
  static async uploadVideo(file, modelSize = 'xs', isLefty = false, proKeypointsFilename = "") {
    // Celery-based API expects POST /videos/upload with query params and file
    const formData = new FormData();
    formData.append('file', file);
    let url = `${API_BASE_URL}/videos/upload?model_size=${modelSize}&is_lefty=${isLefty}`;
    if (proKeypointsFilename) {
      url += `&pro_keypoints_filename=${encodeURIComponent(proKeypointsFilename)}`;
    }
    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ 
          detail: `Upload failed with status ${response.status}` 
        }));
        throw new APIError(
          errorData.detail || 'Upload failed',
          response.status,
          errorData.detail
        );
      }

      // Celery returns { task_id, status, ... }
      return await response.json();
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }
      console.error('Upload failed:', error);
      throw new APIError('Network error during upload');
    }
  }

  /**
   * Process the sample lefty video
   * @param {string} modelSize - Model size to use ('xs', 's', 'm', 'l')
   * @param {string} proKeypointsFilename - Optional pro keypoints filename
   * @returns {Promise<Object>} Upload response with task_id
   */
  static async processSampleLeftyVideo(modelSize = 'xs', proKeypointsFilename = "") {
    // Celery-based API expects POST /videos/sample-lefty with query params
    let url = `${API_BASE_URL}/videos/sample-lefty?model_size=${modelSize}`;
    if (proKeypointsFilename) {
      url += `&pro_keypoints_filename=${encodeURIComponent(proKeypointsFilename)}`;
    }
    try {
      const response = await fetch(url, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ 
          detail: `Sample video processing failed with status ${response.status}` 
        }));
        throw new APIError(
          errorData.detail || 'Sample video processing failed',
          response.status,
          errorData.detail
        );
      }

      // Celery returns { task_id, status, ... }
      return await response.json();
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }
      console.error('Sample video processing failed:', error);
      throw new APIError('Network error during sample video processing');
    }
  }

  /**
   * Get processing status for a job
   * @param {string} taskId - The job ID to check
   * @returns {Promise<Object>} Job status response
   */
  static async getJobStatus(taskId) {
    // Celery-based API: GET /videos/{task_id}/status
    const url = `${API_BASE_URL}/videos/${taskId}/status`;
    try {
      const response = await fetch(url);
      if (!response.ok) {
        if (response.status === 404) {
          throw new APIError('Job not found', 404);
        }
        throw new APIError(`Status check failed with status ${response.status}`, response.status);
      }
      // Celery returns { task_id, status, progress, error, ... }
      return await response.json();
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }
      console.error('Status check failed:', error);
      throw new APIError('Network error during status check');
    }
  }

  /**
   * Get download URL for processed video
   * @param {string} taskId - The job ID
   * @returns {string} Download URL
   */
  static getDownloadUrl(taskId) {
    return `${API_BASE_URL}/videos/${taskId}/download`;
  }

  /**
   * Get preview URL for streaming video in browser
   * @param {string} taskId - The job ID
   * @returns {string} Preview URL
   */
  static getPreviewUrl(taskId) {
    return `${API_BASE_URL}/videos/${taskId}/preview`;
  }

  /**
   * Check if API is healthy
   * @returns {Promise<Object|null>} Health status or null if unhealthy
   */
  static async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        return await response.json();
      }
      return null;
    } catch {
      return null;
    }
  }

  /**
   * Validate file before upload
   * @param {File} file - File to validate
   * @returns {Object} Validation result with isValid boolean and optional error message
   */
  static validateFile(file) {
    // Check file type
    const allowedTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'];
    const allowedExtensions = ['.mp4', '.mov', '.avi', '.mkv'];
    
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
      return {
        isValid: false,
        error: 'Invalid file type. Please upload a video file (.mp4, .mov, .avi, .mkv)'
      };
    }

    // Check file size (100MB limit)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
      return {
        isValid: false,
        error: 'File too large. Maximum size is 100MB'
      };
    }

    return { isValid: true };
  }
}

/**
 * Custom hook for polling job status
 * @param {string|null} taskId - Job ID to poll for
 * @param {Function} onStatusUpdate - Callback function for status updates
 * @param {number} pollingInterval - Polling interval in milliseconds
 * @returns {Object} Object with isPolling boolean and error string
 */
export const useJobPolling = (taskId, onStatusUpdate, pollingInterval = 2000) => {
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState(null);

  const pollStatus = useCallback(async () => {
    if (!taskId) return false;

    try {
      const status = await VideoAPI.getJobStatus(taskId);
      onStatusUpdate(status);
      setError(null);
      
      // Stop polling if job is completed or failed
      if (status.status === 'completed' || status.status === 'failed') {
        return false;
      }
      
      return true; // Continue polling
    } catch (err) {
      const errorMessage = err instanceof APIError ? err.detail || err.message : 'Polling failed';
      console.error('Polling error:', errorMessage);
      setError(errorMessage);
      return false; // Stop polling on error
    }
  }, [taskId, onStatusUpdate]);

  useEffect(() => {
    if (!taskId) {
      setIsPolling(false);
      setError(null);
      return;
    }

    setIsPolling(true);
    setError(null);
    
    let interval;

    // Initial poll
    pollStatus().then(shouldContinue => {
      if (!shouldContinue) {
        setIsPolling(false);
        return;
      }

      // Set up interval polling
      interval = setInterval(async () => {
        const shouldContinue = await pollStatus();
        if (!shouldContinue) {
          clearInterval(interval);
          setIsPolling(false);
        }
      }, pollingInterval);
    });

    // Cleanup function
    return () => {
      if (interval) {
        clearInterval(interval);
      }
      setIsPolling(false);
    };

  }, [taskId, pollStatus, pollingInterval]);

  return { isPolling, error };
};
