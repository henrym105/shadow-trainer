// Shadow Trainer Frontend JavaScript

class ShadowTrainerApp {
    constructor() {
        this.initializeEventListeners();
        this.loadSampleVideos();
    }

    initializeEventListeners() {
        // Input type toggle
        document.querySelectorAll('input[name="inputType"]').forEach(radio => {
            radio.addEventListener('change', this.toggleInputType.bind(this));
        });

        // Process button
        document.getElementById('processBtn').addEventListener('click', this.processVideo.bind(this));
    }

    toggleInputType() {
        const selectedType = document.querySelector('input[name="inputType"]:checked').value;
        const localSection = document.getElementById('localFileSection');
        const s3Section = document.getElementById('s3PathSection');

        if (selectedType === 'local') {
            localSection.style.display = 'block';
            s3Section.style.display = 'none';
        } else {
            localSection.style.display = 'none';
            s3Section.style.display = 'block';
        }
    }

    async processVideo() {
        const inputType = document.querySelector('input[name="inputType"]:checked').value;
        const modelSize = document.getElementById('modelSize').value;
        const handedness = document.querySelector('input[name="handedness"]:checked').value;
        const pitchTypes = this.getSelectedPitchTypes();

        // Validate input
        let file = null;
        if (inputType === 'local') {
            const fileInput = document.getElementById('videoFile');
            if (!fileInput.files.length) {
                this.showError('Please select a video file.');
                return;
            }
            file = fileInput.files[0];
        } else {
            const s3Path = document.getElementById('s3PathInput').value.trim();
            if (!s3Path) {
                this.showError('Please enter an S3 path.');
                return;
            }
            file = s3Path;
        }

        // Show loading state
        this.showLoading();

        try {
            let result;
            if (inputType === 'local') {
                result = await this.processLocalFile(file, modelSize, handedness, pitchTypes);
            } else {
                result = await this.processS3File(file, modelSize, handedness, pitchTypes);
            }

            this.showSuccess(result);
        } catch (error) {
            console.error('Processing error:', error);
            this.showError('Failed to process video: ' + error.message);
        }
    }

    async processLocalFile(file, modelSize, handedness, pitchTypes) {
        // For local files, we need to upload and then process
        const formData = new FormData();
        formData.append('video_file', file);
        formData.append('model_size', modelSize);
        formData.append('handedness', handedness);
        formData.append('pitch_types', pitchTypes.join(','));

        // First upload the file (this would need to be implemented in the backend)
        const uploadResponse = await fetch('/upload_video', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error('Failed to upload video');
        }

        // Then process via API
        const uploadResult = await uploadResponse.json();
        
        // For now, simulate the API call with the uploaded file path
        return await this.callProcessAPI(uploadResult.filename, modelSize, handedness, pitchTypes);
    }

    async processS3File(s3Path, modelSize, handedness, pitchTypes) {
        // Check if it's an asset URL (our sample videos) or actual S3 path
        if (s3Path.startsWith('/assets/') || s3Path.startsWith('/static/')) {
            // For asset URLs, use the frontend S3 processing endpoint
            return await this.processAssetVideo(s3Path, modelSize, handedness, pitchTypes);
        } else {
            // For actual S3 paths, use the API endpoint
            return await this.callProcessAPI(s3Path, modelSize, handedness, pitchTypes);
        }
    }
    
    async processAssetVideo(assetPath, modelSize, handedness, pitchTypes) {
        // Convert asset URL to actual file path for processing
        // For now, we'll pass the asset path directly to the frontend S3 processor
        const formData = new FormData();
        formData.append('s3_path', assetPath);
        formData.append('model_size', modelSize);
        formData.append('handedness', handedness);
        formData.append('pitch_types', pitchTypes.join(','));

        const response = await fetch('/frontend/process_s3_video', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to process sample video');
        }

        return await response.json();
    }

    async callProcessAPI(filePath, modelSize, handedness, pitchTypes) {
        const params = new URLSearchParams({
            file: filePath,
            model_size: modelSize,
            handedness: handedness,
            pitch_type: pitchTypes.join(',')
        });

        const response = await fetch(`/api/v1/process_video/?${params}`, {
            method: 'POST'
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Unknown error occurred');
        }

        return await response.json();
    }

    getSelectedPitchTypes() {
        const checkboxes = document.querySelectorAll('input[type="checkbox"][id^="pitch"]');
        const selected = [];
        checkboxes.forEach(checkbox => {
            if (checkbox.checked) {
                selected.push(checkbox.value);
            }
        });
        return selected;
    }

    showLoading() {
        const resultsSection = document.getElementById('resultsSection');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const videoResult = document.getElementById('videoResult');
        const errorResult = document.getElementById('errorResult');

        resultsSection.style.display = 'block';
        loadingSpinner.style.display = 'block';
        videoResult.style.display = 'none';
        errorResult.style.display = 'none';

        // Add pulse effect to process button
        document.getElementById('processBtn').classList.add('pulse');
        document.getElementById('processBtn').disabled = true;
    }

    showSuccess(result) {
        const loadingSpinner = document.getElementById('loadingSpinner');
        const videoResult = document.getElementById('videoResult');
        const outputVideo = document.getElementById('outputVideo');
        const downloadBtn = document.getElementById('downloadBtn');

        loadingSpinner.style.display = 'none';
        videoResult.style.display = 'block';

        // Set video source
        const videoUrl = result.output_video_local_path || result.output_video_s3_url;
        if (videoUrl) {
            if (videoUrl.startsWith('s3://')) {
                // For S3 URLs, we might need to generate a presigned URL
                outputVideo.src = videoUrl; // This might need backend support
            } else {
                // For local files, serve them through the static file handler
                outputVideo.src = `/static/videos/${videoUrl.split('/').pop()}`;
            }

            // Setup download button
            downloadBtn.onclick = () => {
                const a = document.createElement('a');
                a.href = outputVideo.src;
                a.download = videoUrl.split('/').pop();
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            };
        }

        this.resetProcessButton();
    }

    showError(message) {
        const loadingSpinner = document.getElementById('loadingSpinner');
        const videoResult = document.getElementById('videoResult');
        const errorResult = document.getElementById('errorResult');

        loadingSpinner.style.display = 'none';
        videoResult.style.display = 'none';
        errorResult.style.display = 'block';
        errorResult.textContent = message;

        this.resetProcessButton();
    }

    resetProcessButton() {
        const processBtn = document.getElementById('processBtn');
        processBtn.classList.remove('pulse');
        processBtn.disabled = false;
    }

    async loadSampleVideos() {
        try {
            const sampleVideosContainer = document.getElementById('sampleVideos');
            
            // Show loading state
            sampleVideosContainer.innerHTML = `
                <div class="col-12 text-center">
                    <div class="spinner-border spinner-border-sm" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <span class="ms-2">Loading sample videos...</span>
                </div>
            `;
            
            // Call the backend API to get sample videos
            const response = await fetch('/frontend/sample_videos');
            const data = await response.json();
            
            // Clear loading state
            sampleVideosContainer.innerHTML = '';
            
            if (data.videos && data.videos.length > 0) {
                data.videos.forEach((video, index) => {
                    const col = document.createElement('div');
                    col.className = 'col-md-6 col-lg-4 mb-3';
                    
                    if (video.placeholder) {
                        // Show placeholder for videos that don't exist
                        col.innerHTML = `
                            <div class="card">
                                <div class="card-body text-center">
                                    <i class="fas fa-video fa-3x text-muted mb-3"></i>
                                    <h6 class="card-title">${video.name}</h6>
                                    <p class="text-muted small">Sample video placeholder</p>
                                    <button class="btn btn-outline-primary btn-sm" disabled>
                                        <i class="fas fa-upload me-1"></i>Upload Your Own
                                    </button>
                                </div>
                            </div>
                        `;
                    } else {
                        // Show actual video
                        col.innerHTML = `
                            <div class="card">
                                <video controls class="card-img-top" style="height: 200px; object-fit: cover;">
                                    <source src="${video.url}" type="video/mp4">
                                    <source src="${video.url}" type="video/mov">
                                    Your browser does not support the video tag.
                                </video>
                                <div class="card-body">
                                    <h6 class="card-title">${video.name}</h6>
                                    <button class="btn btn-primary btn-sm use-sample-video-btn" data-video-url="${video.url}" data-video-name="${video.name}">
                                        <i class="fas fa-play me-1"></i>Use This Video
                                    </button>
                                </div>
                            </div>
                        `;
                    }
                    
                    sampleVideosContainer.appendChild(col);
                });
                
                // Add event listeners to all "Use This Video" buttons
                const useSampleBtns = sampleVideosContainer.querySelectorAll('.use-sample-video-btn');
                console.log('Found sample video buttons:', useSampleBtns.length);
                useSampleBtns.forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        console.log('Sample video button clicked!');
                        // Make sure we get the button element, not a child element
                        const button = e.currentTarget;
                        const videoUrl = button.getAttribute('data-video-url');
                        const videoName = button.getAttribute('data-video-name');
                        console.log('Video URL:', videoUrl, 'Video Name:', videoName);
                        this.useSampleVideo(videoUrl, videoName);
                    });
                });
            } else {
                // No videos found
                sampleVideosContainer.innerHTML = `
                    <div class="col-12 text-center">
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i>
                            No sample videos available. Upload your own video to get started!
                        </div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Failed to load sample videos:', error);
            
            // Show error state
            const sampleVideosContainer = document.getElementById('sampleVideos');
            sampleVideosContainer.innerHTML = `
                <div class="col-12 text-center">
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        Unable to load sample videos. You can still upload your own video above.
                    </div>
                </div>
            `;
        }
    }
    
    useSampleVideo(videoUrl, videoName) {
        console.log('useSampleVideo called with:', videoUrl, videoName);
        // Switch to S3 path input mode
        const s3Radio = document.getElementById('s3Path');
        const localRadio = document.getElementById('localFile');
        
        if (s3Radio) {
            s3Radio.checked = true;
            this.toggleInputType(); // This will show the S3 section and hide local file section
        }
        
        // Populate the S3 path input with the video URL
        const s3PathInput = document.getElementById('s3PathInput');
        if (s3PathInput) {
            s3PathInput.value = videoUrl;
        }
        
        // Scroll to the input section
        const inputSection = document.querySelector('.card');
        if (inputSection) {
            inputSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        
        // Show a success message
        this.showNotification(`Selected sample video: ${videoName}`, 'success');
        
        // Highlight the S3 input to draw attention
        if (s3PathInput) {
            s3PathInput.style.borderColor = '#28a745';
            s3PathInput.style.boxShadow = '0 0 0 0.2rem rgba(40, 167, 69, 0.25)';
            
            // Remove the highlight after 3 seconds
            setTimeout(() => {
                s3PathInput.style.borderColor = '';
                s3PathInput.style.boxShadow = '';
            }, 3000);
        }
    }
    
    showNotification(message, type = 'info') {
        // Create a notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = `
            top: 20px;
            right: 20px;
            z-index: 1050;
            max-width: 300px;
        `;
        
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ShadowTrainerApp();
});
