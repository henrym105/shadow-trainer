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
        return await this.callProcessAPI(s3Path, modelSize, handedness, pitchTypes);
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
            // This would load sample videos from the videos directory
            const sampleVideosContainer = document.getElementById('sampleVideos');
            
            // For now, create placeholder sample videos
            const sampleVideos = [
                { name: 'Henry1-mini.mov', path: '/static/videos/Henry1-mini.mov' },
                { name: 'pitch_sample_4.mp4', path: '/static/videos/pitch_sample_4.mp4' }
            ];

            sampleVideos.forEach((video, index) => {
                const col = document.createElement('div');
                col.className = 'col-md-6 sample-video-item';
                
                col.innerHTML = `
                    <video controls class="w-100">
                        <source src="${video.path}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <div class="sample-video-caption">${video.name}</div>
                `;
                
                sampleVideosContainer.appendChild(col);
            });
        } catch (error) {
            console.error('Failed to load sample videos:', error);
        }
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ShadowTrainerApp();
});
