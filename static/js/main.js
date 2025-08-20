// Main JavaScript for YOLOv8 Training Dashboard

// Global variables
let trainingStatusInterval;
let elapsedTimeInterval;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    console.log('YOLOv8 Training Dashboard initialized');
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize file upload handlers
    initializeFileUploads();
    
    // Initialize training status monitoring
    initializeTrainingMonitoring();
    
    // Initialize form validations
    initializeFormValidations();
    
    // Initialize auto-save functionality
    initializeAutoSave();
}

// Tooltip initialization
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// File upload enhancements
function initializeFileUploads() {
    // Add drag and drop functionality to file inputs
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        const parentElement = input.closest('.mb-3') || input.parentElement;
        
        // Drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            parentElement.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            parentElement.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            parentElement.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight(e) {
            parentElement.classList.add('border-primary');
        }
        
        function unhighlight(e) {
            parentElement.classList.remove('border-primary');
        }
        
        // Handle dropped files
        parentElement.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (input.multiple) {
                input.files = files;
            } else if (files.length > 0) {
                const fileArray = Array.from(files);
                const newFileList = new DataTransfer();
                newFileList.items.add(fileArray[0]);
                input.files = newFileList.files;
            }
            
            // Trigger change event
            input.dispatchEvent(new Event('change', { bubbles: true }));
        }
        
        // File validation and preview
        input.addEventListener('change', function(e) {
            validateFiles(this);
            showFilePreview(this);
        });
    });
}

// File validation
function validateFiles(input) {
    const files = Array.from(input.files);
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    
    let validFiles = [];
    let errors = [];
    
    files.forEach(file => {
        if (file.size > maxSize) {
            errors.push(`${file.name} is too large (max 10MB)`);
        } else if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.zip')) {
            errors.push(`${file.name} is not a supported file type`);
        } else {
            validFiles.push(file);
        }
    });
    
    if (errors.length > 0) {
        showNotification('File Validation Errors', errors.join('<br>'), 'danger');
    }
    
    return validFiles.length > 0;
}

// File preview
function showFilePreview(input) {
    const files = Array.from(input.files);
    const previewContainer = input.closest('.tab-pane').querySelector('.file-preview');
    
    if (previewContainer) {
        previewContainer.innerHTML = '';
        
        files.forEach(file => {
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.className = 'img-thumbnail me-2 mb-2';
                    img.style.maxWidth = '100px';
                    img.style.maxHeight = '100px';
                    previewContainer.appendChild(img);
                };
                reader.readAsDataURL(file);
            }
        });
    }
}

// Training status monitoring
function initializeTrainingMonitoring() {
    // Check if we're on a page that needs training monitoring
    const trainingStatusElement = document.querySelector('[data-training-status]');
    
    if (trainingStatusElement) {
        startTrainingMonitoring();
    }
}

function startTrainingMonitoring() {
    // Update training status every 5 seconds
    trainingStatusInterval = setInterval(updateTrainingStatus, 5000);
    
    // Update elapsed time every second
    elapsedTimeInterval = setInterval(updateElapsedTime, 1000);
}

function stopTrainingMonitoring() {
    if (trainingStatusInterval) {
        clearInterval(trainingStatusInterval);
        trainingStatusInterval = null;
    }
    
    if (elapsedTimeInterval) {
        clearInterval(elapsedTimeInterval);
        elapsedTimeInterval = null;
    }
}

function updateTrainingStatus() {
    fetch('/api/training_status')
        .then(response => response.json())
        .then(data => {
            updateTrainingUI(data);
        })
        .catch(error => {
            console.error('Error fetching training status:', error);
        });
}

function updateTrainingUI(status) {
    // Update progress bars
    const progressBars = document.querySelectorAll('.training-progress');
    progressBars.forEach(bar => {
        const progress = status.total_epochs > 0 ? 
            (status.current_epoch / status.total_epochs * 100) : 0;
        bar.style.width = `${progress}%`;
        bar.textContent = `${status.current_epoch} / ${status.total_epochs}`;
    });
    
    // Update metrics
    updateMetricElement('current-epoch', status.current_epoch);
    updateMetricElement('total-epochs', status.total_epochs);
    updateMetricElement('current-loss', status.loss.toFixed(4));
    updateMetricElement('current-accuracy', (status.accuracy * 100).toFixed(2) + '%');
    updateMetricElement('training-status', status.status);
    
    // Update training state indicators
    const trainingIndicators = document.querySelectorAll('.training-indicator');
    trainingIndicators.forEach(indicator => {
        if (status.is_training) {
            indicator.classList.add('status-training');
            indicator.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training';
        } else {
            indicator.classList.remove('status-training');
            indicator.innerHTML = '<i class="fas fa-pause"></i> ' + status.status;
        }
    });
}

function updateMetricElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function updateElapsedTime() {
    const elapsedElement = document.getElementById('elapsed-time');
    if (elapsedElement && window.trainingStartTime) {
        const now = new Date();
        const start = new Date(window.trainingStartTime);
        const elapsed = Math.floor((now - start) / 1000);
        
        const hours = Math.floor(elapsed / 3600);
        const minutes = Math.floor((elapsed % 3600) / 60);
        const seconds = elapsed % 60;
        
        const timeString = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        elapsedElement.textContent = timeString;
    }
}

// Form validations
function initializeFormValidations() {
    const forms = document.querySelectorAll('form[data-validate]');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(this)) {
                e.preventDefault();
                e.stopPropagation();
            }
            this.classList.add('was-validated');
        });
    });
}

function validateForm(form) {
    let isValid = true;
    
    // Validate required fields
    const requiredFields = form.querySelectorAll('[required]');
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            isValid = false;
            showFieldError(field, 'This field is required');
        } else {
            clearFieldError(field);
        }
    });
    
    // Validate number ranges
    const numberFields = form.querySelectorAll('input[type="number"]');
    numberFields.forEach(field => {
        const value = parseFloat(field.value);
        const min = parseFloat(field.getAttribute('min'));
        const max = parseFloat(field.getAttribute('max'));
        
        if (!isNaN(min) && value < min) {
            isValid = false;
            showFieldError(field, `Value must be at least ${min}`);
        } else if (!isNaN(max) && value > max) {
            isValid = false;
            showFieldError(field, `Value must be no more than ${max}`);
        } else {
            clearFieldError(field);
        }
    });
    
    return isValid;
}

function showFieldError(field, message) {
    clearFieldError(field);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    
    field.classList.add('is-invalid');
    field.parentNode.appendChild(errorDiv);
}

function clearFieldError(field) {
    field.classList.remove('is-invalid');
    const errorDiv = field.parentNode.querySelector('.invalid-feedback');
    if (errorDiv) {
        errorDiv.remove();
    }
}

// Auto-save functionality
function initializeAutoSave() {
    const autoSaveForms = document.querySelectorAll('[data-auto-save]');
    
    autoSaveForms.forEach(form => {
        const inputs = form.querySelectorAll('input, select, textarea');
        
        inputs.forEach(input => {
            input.addEventListener('change', function() {
                debounce(saveFormData, 1000)(form);
            });
        });
    });
}

function saveFormData(form) {
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    localStorage.setItem(`autosave_${form.id}`, JSON.stringify(data));
    showNotification('Auto-saved', 'Form data has been automatically saved', 'info', 2000);
}

function loadFormData(form) {
    const savedData = localStorage.getItem(`autosave_${form.id}`);
    
    if (savedData) {
        const data = JSON.parse(savedData);
        
        Object.keys(data).forEach(key => {
            const field = form.querySelector(`[name="${key}"]`);
            if (field) {
                if (field.type === 'checkbox') {
                    field.checked = data[key] === 'on';
                } else {
                    field.value = data[key];
                }
            }
        });
    }
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function showNotification(title, message, type = 'info', duration = 5000) {
    const toastContainer = getOrCreateToastContainer();
    
    const toast = document.createElement('div');
    toast.className = 'toast show';
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="toast-header">
            <i class="fas fa-${getIconForType(type)} me-2 text-${type}"></i>
            <strong class="me-auto">${title}</strong>
            <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, duration);
    
    // Initialize Bootstrap toast
    new bootstrap.Toast(toast);
}

function getOrCreateToastContainer() {
    let container = document.querySelector('.toast-container');
    
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(container);
    }
    
    return container;
}

function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'info': 'info-circle',
        'warning': 'exclamation-triangle',
        'danger': 'exclamation-circle',
        'error': 'exclamation-circle'
    };
    
    return icons[type] || 'info-circle';
}

function showLoadingOverlay(message = 'Loading...') {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="text-center">
            <div class="loading-spinner"></div>
            <p class="mt-3 text-white">${message}</p>
        </div>
    `;
    
    document.body.appendChild(overlay);
    return overlay;
}

function hideLoadingOverlay(overlay) {
    if (overlay && overlay.parentNode) {
        overlay.parentNode.removeChild(overlay);
    }
}

// Export functions for global use
window.YOLOApp = {
    showNotification,
    showLoadingOverlay,
    hideLoadingOverlay,
    updateTrainingStatus,
    startTrainingMonitoring,
    stopTrainingMonitoring
};
