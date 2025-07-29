import React, { useState, useRef } from 'react';
import './FileUpload.css';
import './FileUpload.css';

const FileUpload = ({ 
  selectedFile, 
  onFileSelect, 
  uploadError, 
  disabled = false 
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);

  const maxSize = 15 * 1024 * 1024; // 15MB

  const validateFile = (file) => {
    const validTypes = ['.mp4', '.mov', '.avi', '.mkv'];

    
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
        error: `File size too large. Please upload a video smaller than ${maxSize / (1024 * 1024)}MB` 
      };
    }

    return { valid: true };
  };

  const handleFileSelect = (file) => {
    const validation = validateFile(file);
    
    if (!validation.valid) {
      onFileSelect(null, validation.error);
      return;
    }

    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    
    onFileSelect(file, null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    if (disabled) return;
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragOver(true);
    }
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleFileInputChange = (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleClick = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const clearSelection = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    onFileSelect(null, null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

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
        className={`file-upload-zone ${isDragOver ? 'drag-over' : ''} ${disabled ? 'disabled' : ''} ${selectedFile ? 'has-file' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".mp4,.mov,.avi,.mkv,video/*"
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
          disabled={disabled}
        />
        
        {!selectedFile ? (
          <div className="upload-prompt">
            <div className="upload-icon">📹</div>
            <h3>Drop your video here</h3>
            <p>or click to browse</p>
            <div className="file-requirements">
              <p>Supported formats: MP4, MOV, AVI, MKV</p>
              <p>Maximum size: {maxSize / (1024 * 1024)}MB</p>
            </div>
          </div>
        ) : (
          <div className="file-info">
            <div className="file-details">
              <h4>{selectedFile.name}</h4>
              <p>Size: {formatFileSize(selectedFile.size)}</p>
              <button 
                type="button" 
                className="clear-button"
                onClick={(e) => {
                  e.stopPropagation();
                  clearSelection();
                }}
              >
                Clear Selection
              </button>
            </div>
            {previewUrl && (
              <div className="video-preview">
                <video
                  src={previewUrl}
                  controls
                  width="200"
                  height="150"
                  onError={() => setPreviewUrl(null)}
                />
              </div>
            )}
          </div>
        )}
      </div>
      
      {uploadError && (
        <div className="upload-error">
          <span className="error-icon">⚠️</span>
          {uploadError}
        </div>
      )}
    </div>
  );
};

export default FileUpload;
