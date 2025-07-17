/**
 * File Upload Component
 * Handles video file selection and validation
 */

import React, { useState, useRef } from 'react';
import './FileUpload.css';

const FileUpload = ({ onFileSelect, disabled = false, acceptedFormats = ['.mp4', '.mov', '.avi', '.mkv'] }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  // Validate file type and size
  const validateFile = (file) => {
    if (!file) return { valid: false, error: 'No file selected' };

    // Check file extension
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    if (!acceptedFormats.includes(fileExtension)) {
      return { 
        valid: false, 
        error: `Invalid file type. Accepted formats: ${acceptedFormats.join(', ')}` 
      };
    }

    // Check file size (max 100MB)
    const maxSize = 100 * 1024 * 1024; // 100MB in bytes
    if (file.size > maxSize) {
      return { 
        valid: false, 
        error: 'File too large. Maximum size is 100MB.' 
      };
    }

    return { valid: true, error: null };
  };

  // Handle file selection
  const handleFileSelect = (file) => {
    const validation = validateFile(file);
    
    if (validation.valid) {
      setSelectedFile(file);
      onFileSelect(file, null);
    } else {
      setSelectedFile(null);
      onFileSelect(null, validation.error);
    }
  };

  // Handle drag events
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragIn = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setDragActive(true);
    }
  };

  const handleDragOut = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  // Handle input change
  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  // Handle click to open file dialog
  const handleClick = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Clear selection
  const handleClear = () => {
    setSelectedFile(null);
    onFileSelect(null, null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="file-upload-container">
      <div
        className={`file-upload-dropzone ${dragActive ? 'drag-active' : ''} ${disabled ? 'disabled' : ''} ${selectedFile ? 'has-file' : ''}`}
        onDragEnter={handleDragIn}
        onDragLeave={handleDragOut}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={!selectedFile ? handleClick : undefined}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedFormats.join(',')}
          onChange={handleInputChange}
          disabled={disabled}
          style={{ display: 'none' }}
        />
        {selectedFile ? (
          <div className="file-selected">
            <div className="file-icon">🎥</div>
            <div className="file-info">
              <div className="file-name">{selectedFile.name}</div>
              <div className="file-size">{formatFileSize(selectedFile.size)}</div>
            </div>
            <button 
              className="clear-button"
              onClick={(e) => {
                e.stopPropagation();
                handleClear();
              }}
              disabled={disabled}
            >
              ✕
            </button>
          </div>
        ) : (
          <div className="file-upload-content">
            <div className="upload-icon">📁</div>
            <div className="upload-text">
              <strong>Click to upload</strong> or drag and drop
            </div>
            <div className="upload-formats">
              Supported formats: {acceptedFormats.join(', ')}
            </div>
            <div className="upload-size">
              Maximum file size: 100MB
            </div>
          </div>
        )}
      </div>
      {/* Video preview below the green box, outside the dropzone */}
      {selectedFile && (
        <div className="file-preview" style={{ marginTop: '1rem', width: '100%' }}>
          <video
            src={URL.createObjectURL(selectedFile)}
            controls
            style={{ width: '100%', maxHeight: '320px', borderRadius: '12px', boxShadow: '0 2px 12px rgba(0,0,0,0.08)' }}
          />
        </div>
      )}
    </div>
  );
};

export default FileUpload;
