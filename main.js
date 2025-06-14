// Check if user is authenticated
function checkAuth() {
    const authMessage = document.querySelector('.auth-message');
    if (authMessage) {
        // User is not authenticated, redirect to login
        window.location.href = '/login';
        return false;
    }
    return true;
}

// Handle file upload only if authenticated
function handleFileUpload(file) {
    if (!checkAuth()) return;
    
    // Rest of the file upload handling code...
}

document.addEventListener('DOMContentLoaded', () => {
    // Only initialize if authenticated
    if (!checkAuth()) return;
    
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const mediaPreview = document.getElementById('mediaPreview');
    const resultsSection = document.getElementById('resultsSection');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceScore = document.getElementById('confidenceScore');
    const resultText = document.getElementById('resultText');
    const analysisTime = document.getElementById('analysisTime');

    let currentFile = null;
    let currentAnalysisId = null;

    // Stats Elements
    const totalAnalyses = document.querySelector('.stat-card:nth-child(1) .stat-value');
    const successfulAnalyses = document.querySelector('.stat-card:nth-child(2) .stat-value');
    const averageConfidence = document.querySelector('.stat-card:nth-child(3) .stat-value');

    // Initialize stats
    let stats = {
        totalAnalyses: 0,
        successfulAnalyses: 0,
        totalConfidence: 0
    };

    // Update stats display
    function updateStats() {
        if (totalAnalyses) totalAnalyses.textContent = stats.totalAnalyses;
        if (successfulAnalyses) successfulAnalyses.textContent = stats.successfulAnalyses;
        if (averageConfidence) {
            const avgConf = stats.successfulAnalyses > 0 ? 
                Math.round(stats.totalConfidence / stats.successfulAnalyses) : 0;
            averageConfidence.textContent = `${avgConf}%`;
        }
    }

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('highlight');
    }

    function unhighlight(e) {
        dropZone.classList.remove('highlight');
    }

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        const file = files[0];
        if (!file) return;

        // Check file type
        const fileType = file.type.split('/')[0];
        if (fileType !== 'image' && fileType !== 'video') {
            alert('Please upload an image or video file');
            return;
        }

        // Check file size (max 16MB)
        const maxSize = 16 * 1024 * 1024; // 16MB in bytes
        if (file.size > maxSize) {
            alert('File size exceeds 16MB limit');
            return;
        }

        // Show preview
        showPreview(file);
        
        // Update stats
        stats.totalAnalyses++;
        updateStats();
    }

    function showPreview(file) {
        const isImage = file.type.startsWith('image/');
        const isVideo = file.type.startsWith('video/');

        // Clear previous preview
        mediaPreview.innerHTML = '';

        if (isImage) {
            const img = document.createElement('img');
            img.style.maxWidth = '100%';
            img.style.maxHeight = '300px';
            img.style.objectFit = 'contain';
            img.src = URL.createObjectURL(file);
            mediaPreview.appendChild(img);
        } else if (isVideo) {
            const video = document.createElement('video');
            video.controls = true;
            video.style.maxWidth = '100%';
            video.style.maxHeight = '300px';
            video.src = URL.createObjectURL(file);
            mediaPreview.appendChild(video);
        }

        // Show results section
        resultsSection.style.display = 'block';
        
        // Start analysis
        analyzeMedia(file);
    }

    async function analyzeMedia(file) {
        if (!file) return;

        loadingOverlay.style.display = 'flex';
        const startTime = new Date();

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                // Store the analysis ID for report download
                if (data.analysis_id) {
                    currentAnalysisId = data.analysis_id;
                    console.log('Stored analysis ID:', currentAnalysisId);
                }

                // Update confidence bar
                const confidence = Math.round(data.confidence * 100);
                confidenceBar.style.width = `${confidence}%`;
                confidenceScore.textContent = `${confidence}%`;

                // Update result text with enhanced styling
                resultText.textContent = data.result;
                // Change color based on result and confidence
                if (data.confidence > 0.5 && data.result.includes('Autism Detected')) {
                    resultText.className = 'negative'; // Red pulsing effect for autism detected with high confidence
                    resultText.style.fontWeight = '700'; // Make text bolder
                } else if (data.result.includes('No Autism Detected')) {
                    resultText.className = 'positive'; // Green for no autism
                    resultText.style.fontWeight = '600';
                } else {
                    resultText.className = data.confidence > 0.5 ? 'positive' : 'negative';
                    resultText.style.fontWeight = '600';
                }

                // Update stats
                stats.successfulAnalyses++;
                stats.totalConfidence += confidence;
                updateStats();

                // Update analysis time
                const endTime = new Date();
                const timeDiff = (endTime - startTime) / 1000;
                analysisTime.textContent = `${timeDiff.toFixed(2)}s`;

                // Show results section if hidden
                const resultsSection = document.getElementById('resultsSection');
                resultsSection.style.display = 'block';
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Error during analysis:', error);
            resultText.textContent = 'Error during analysis';
            resultText.className = 'error';
            confidenceBar.style.width = '0%';
            confidenceScore.textContent = '0%';
        } finally {
            loadingOverlay.style.display = 'none';
        }
    }

    // Initialize
    updateStats();
});

// Download report function
async function downloadReport() {
    if (!checkAuth()) return;
    
    if (!currentAnalysisId) {
        alert('No analysis available to download');
        return;
    }

    try {
        // Show loading overlay
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'flex';
        }

        // Fetch the report from the server
        const response = await fetch(`/download_report/${currentAnalysisId}`);
        
        if (!response.ok) {
            throw new Error('Failed to generate report');
        }

        // Get the blob from the response
        const blob = await response.blob();
        
        // Create a download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `autism_analysis_report_${currentAnalysisId}.pdf`;
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch (error) {
        console.error('Error downloading report:', error);
        alert('Error downloading report: ' + error.message);
    } finally {
        // Hide loading overlay
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }
}

// Share results function
function shareResults() {
    if (!checkAuth()) return;
    
    const resultText = document.getElementById('resultText').textContent;
    const confidenceScore = document.getElementById('confidenceScore').textContent;
    
    const shareText = `Autism Detection Analysis Result:\n${resultText}\nConfidence Score: ${confidenceScore}`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Autism Detection Analysis',
            text: shareText
        }).catch(console.error);
    } else {
        alert('Sharing is not supported on this browser');
    }
} 