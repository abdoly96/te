import os
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import zipfile
import glob
from utils.file_handler import FileHandler
from utils.image_processor import ImageProcessor
from utils.model_trainer import ModelTrainer
from utils.enhanced_trainer import EnhancedYOLOTrainer
from utils.dataset_manager import DatasetManager

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "your-secret-key-here")

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
MODELS_FOLDER = 'models'
CONFIG_FILE = 'config.json'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Initialize utilities
file_handler = FileHandler()
image_processor = ImageProcessor()
model_trainer = ModelTrainer()
enhanced_trainer = EnhancedYOLOTrainer()
dataset_manager = DatasetManager()

# Import API routes
from api_routes import create_api_routes

# Create API routes
create_api_routes(app, enhanced_trainer, file_handler, image_processor, dataset_manager)

# Global training state
training_state = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'loss': 0.0,
    'accuracy': 0.0,
    'progress': 0,
    'status': 'Ready',
    'start_time': None,
    'logs': []
}

def load_config():
    """Load configuration from JSON file"""
    default_config = {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001,
        'image_size': 640,
        'model_path': 'models/',
        'data_path': 'processed/',
        'auto_save': True,
        'checkpoint_interval': 10
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                default_config.update(config)
        except Exception as e:
            logging.error(f"Error loading config: {e}")
    
    return default_config

def save_config(config):
    """Save configuration to JSON file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        logging.error(f"Error saving config: {e}")
        return False

def get_next_model_name():
    """Generate next model name in sequence (yolo1.pt, yolo2.pt, etc.)"""
    existing_models = glob.glob(os.path.join(MODELS_FOLDER, 'yolo*.pt'))
    if not existing_models:
        return 'yolo1.pt'
    
    numbers = []
    for model in existing_models:
        basename = os.path.basename(model)
        try:
            # Extract number from yolo{number}.pt
            number = int(basename.replace('yolo', '').replace('.pt', ''))
            numbers.append(number)
        except ValueError:
            continue
    
    next_number = max(numbers) + 1 if numbers else 1
    return f'yolo{next_number}.pt'

@app.route('/')
def dashboard():
    """Dashboard page showing training status and metrics"""
    config = load_config()
    
    # Get recent models
    recent_models = []
    model_files = glob.glob(os.path.join(MODELS_FOLDER, '*.pt'))
    for model_file in sorted(model_files, key=os.path.getmtime, reverse=True)[:5]:
        stat = os.stat(model_file)
        recent_models.append({
            'name': os.path.basename(model_file),
            'size': f"{stat.st_size / (1024*1024):.1f} MB",
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
        })
    
    # Get training history (mock data for now)
    training_history = {
        'epochs': list(range(1, training_state['current_epoch'] + 1)) if training_state['current_epoch'] > 0 else [1],
        'loss': [0.5 - (i * 0.01) for i in range(training_state['current_epoch'])] if training_state['current_epoch'] > 0 else [0.5],
        'accuracy': [0.7 + (i * 0.005) for i in range(training_state['current_epoch'])] if training_state['current_epoch'] > 0 else [0.7]
    }
    
    return render_template('dashboard.html', 
                         training_state=training_state,
                         recent_models=recent_models,
                         training_history=training_history,
                         config=config)

@app.route('/training')
def training():
    """Training page for uploading data and starting training"""
    config = load_config()
    
    # Get uploaded files count
    uploaded_files = len([f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))])
    processed_files = len([f for f in os.listdir(PROCESSED_FOLDER) if os.path.isfile(os.path.join(PROCESSED_FOLDER, f))])
    
    return render_template('training.html', 
                         config=config,
                         uploaded_files=uploaded_files,
                         processed_files=processed_files,
                         training_state=training_state)

@app.route('/settings')
def settings():
    """Settings page for configuring hyperparameters"""
    config = load_config()
    return render_template('settings.html', config=config)

@app.route('/save_settings', methods=['POST'])
def save_settings():
    """Save settings configuration"""
    try:
        config = load_config()
        
        # Update config with form data
        config['epochs'] = int(request.form.get('epochs', 100))
        config['batch_size'] = int(request.form.get('batch_size', 16))
        config['learning_rate'] = float(request.form.get('learning_rate', 0.001))
        config['image_size'] = int(request.form.get('image_size', 640))
        config['auto_save'] = 'auto_save' in request.form
        config['checkpoint_interval'] = int(request.form.get('checkpoint_interval', 10))
        
        # Save updated config
        if save_config(config):
            flash('تم حفظ الإعدادات بنجاح', 'success')
        else:
            flash('فشل في حفظ الإعدادات', 'error')
            
    except Exception as e:
        logging.error(f"Error saving settings: {e}")
        flash('حدث خطأ أثناء حفظ الإعدادات', 'error')
    
    return redirect(url_for('settings'))

@app.route('/data')
def data_management():
    """Data management page for viewing and managing uploaded data"""
    # Get uploaded files
    uploaded_files = []
    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            uploaded_files.append({
                'name': filename,
                'size': f"{stat.st_size / 1024:.1f} KB",
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                'type': 'Image' if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) else 'Other'
            })
    
    # Get processed files
    processed_files = []
    for filename in os.listdir(PROCESSED_FOLDER):
        filepath = os.path.join(PROCESSED_FOLDER, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            processed_files.append({
                'name': filename,
                'size': f"{stat.st_size / 1024:.1f} KB",
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    return render_template('data.html', 
                         uploaded_files=uploaded_files,
                         processed_files=processed_files)

@app.route('/annotation')
def annotation():
    """Image annotation page for creating bounding box labels"""
    return render_template('annotation.html')

@app.route('/inference', methods=['GET', 'POST'])
def inference():
    """Model inference page for testing trained models"""
    # Get available trained models
    models_dir = Path('models')
    available_models = []
    
    if models_dir.exists():
        for model_file in models_dir.glob('*.pt'):
            stat = model_file.stat()
            available_models.append({
                'name': model_file.name,
                'path': str(model_file),
                'date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                'size': f"{stat.st_size / 1024 / 1024:.1f} MB"
            })
    
    if request.method == 'POST':
        # Handle image upload and inference
        try:
            if 'images' not in request.files:
                flash('No images uploaded', 'error')
                return redirect(url_for('inference'))
            
            files = request.files.getlist('images')
            model_path = request.form.get('model_path')
            confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
            
            if not model_path:
                flash('No model selected', 'error')
                return redirect(url_for('inference'))
            
            # Process uploaded images
            results = []
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)
                    
                    # Run inference using the enhanced trainer
                    try:
                        inference_result = enhanced_trainer.run_inference(filepath, confidence_threshold)
                        results.append({
                            'filename': filename,
                            'detections': inference_result.get('detections', []),
                            'image_data': inference_result.get('image_data', ''),
                            'processing_time': inference_result.get('processing_time', 0),
                            'avg_confidence': inference_result.get('avg_confidence', 0)
                        })
                    except Exception as e:
                        logging.error(f"Inference error for {filename}: {e}")
                        results.append({
                            'filename': filename,
                            'detections': [],
                            'image_data': '',
                            'processing_time': 0,
                            'avg_confidence': 0
                        })
            
            # Calculate performance stats
            if results:
                total_images = len(results)
                total_detections = sum(len(r['detections']) for r in results)
                total_time = sum(r['processing_time'] for r in results)
                avg_confidence = sum(r['avg_confidence'] for r in results) / total_images if total_images > 0 else 0
                
                performance_stats = {
                    'total_images': total_images,
                    'total_detections': total_detections,
                    'total_time': total_time,
                    'avg_confidence': avg_confidence
                }
            else:
                performance_stats = None
            
            return render_template('inference.html', 
                                available_models=available_models,
                                results=results,
                                performance_stats=performance_stats,
                                selected_model=model_path)
            
        except Exception as e:
            logging.error(f"Inference error: {e}")
            flash(f'Inference failed: {str(e)}', 'error')
            return redirect(url_for('inference'))
    
    # GET method - show the form
    return render_template('inference.html', available_models=available_models)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        upload_type = request.form.get('upload_type')
        
        if upload_type == 'single':
            file = request.files.get('file')
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                
                # Validate and process image
                if image_processor.validate_image(filepath):
                    processed_path = os.path.join(PROCESSED_FOLDER, filename)
                    if image_processor.process_image(filepath, processed_path):
                        flash(f'Successfully uploaded and processed {filename}', 'success')
                    else:
                        flash(f'Uploaded {filename} but processing failed', 'warning')
                else:
                    flash(f'Invalid image file: {filename}', 'error')
                    os.remove(filepath)
        
        elif upload_type == 'multiple':
            files = request.files.getlist('files')
            processed_count = 0
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)
                    
                    if image_processor.validate_image(filepath):
                        processed_path = os.path.join(PROCESSED_FOLDER, filename)
                        if image_processor.process_image(filepath, processed_path):
                            processed_count += 1
                        else:
                            os.remove(filepath)
                    else:
                        os.remove(filepath)
            
            flash(f'Successfully processed {processed_count} out of {len(files)} files', 'success')
        
        elif upload_type == 'zip':
            zip_file = request.files.get('zip_file')
            if zip_file and zip_file.filename and zip_file.filename.endswith('.zip'):
                zip_path = os.path.join(UPLOAD_FOLDER, secure_filename(zip_file.filename))
                zip_file.save(zip_path)
                
                # Extract ZIP file
                extracted_count = file_handler.extract_zip(zip_path, UPLOAD_FOLDER)
                os.remove(zip_path)  # Remove ZIP after extraction
                
                # Process extracted images
                processed_count = 0
                for filename in os.listdir(UPLOAD_FOLDER):
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    if os.path.isfile(filepath) and image_processor.validate_image(filepath):
                        processed_path = os.path.join(PROCESSED_FOLDER, filename)
                        if image_processor.process_image(filepath, processed_path):
                            processed_count += 1
                
                flash(f'Extracted {extracted_count} files, processed {processed_count} images', 'success')
    
    except Exception as e:
        logging.error(f"Upload error: {e}")
        flash(f'Upload failed: {str(e)}', 'error')
    
    return redirect(url_for('training'))

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start model training"""
    if training_state['is_training']:
        flash('Training is already in progress', 'warning')
        return redirect(url_for('training'))
    
    try:
        config = load_config()
        training_type = request.form.get('training_type', 'new')
        
        # Check if we have processed data
        processed_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not processed_files:
            flash('No processed images found. Please upload and process images first.', 'error')
            return redirect(url_for('training'))
        
        # Start training in a separate thread
        def train_model():
            global training_state
            try:
                training_state['is_training'] = True
                training_state['status'] = 'Initializing...'
                training_state['start_time'] = datetime.now()
                training_state['total_epochs'] = config['epochs']
                training_state['current_epoch'] = 0
                training_state['logs'] = []
                
                # Simulate training process (replace with actual YOLOv8 training)
                model_trainer.train_yolo_model(
                    data_path=PROCESSED_FOLDER,
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    learning_rate=config['learning_rate'],
                    image_size=config['image_size'],
                    training_state=training_state,
                    resume=(training_type == 'resume')
                )
                
                # Save model with auto-generated name
                model_name = get_next_model_name()
                model_path = os.path.join(MODELS_FOLDER, model_name)
                
                training_state['status'] = f'Training completed. Model saved as {model_name}'
                training_state['is_training'] = False
                
            except Exception as e:
                logging.error(f"Training error: {e}")
                training_state['status'] = f'Training failed: {str(e)}'
                training_state['is_training'] = False
        
        thread = threading.Thread(target=train_model)
        thread.daemon = True
        thread.start()
        
        flash('Training started successfully', 'success')
        
    except Exception as e:
        logging.error(f"Start training error: {e}")
        flash(f'Failed to start training: {str(e)}', 'error')
    
    return redirect(url_for('training'))

@app.route('/stop_training', methods=['POST'])
def stop_training():
    """Stop current training"""
    if training_state['is_training']:
        training_state['is_training'] = False
        training_state['status'] = 'Training stopped by user'
        flash('Training stopped', 'info')
    else:
        flash('No training in progress', 'warning')
    
    return redirect(url_for('training'))



@app.route('/delete_file/<path:filename>', methods=['POST'])
def delete_file(filename):
    """Delete uploaded or processed file"""
    try:
        file_type = request.form.get('file_type', 'uploaded')
        
        if file_type == 'uploaded':
            filepath = os.path.join(UPLOAD_FOLDER, filename)
        else:
            filepath = os.path.join(PROCESSED_FOLDER, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            flash(f'Successfully deleted {filename}', 'success')
        else:
            flash(f'File {filename} not found', 'error')
    
    except Exception as e:
        logging.error(f"Delete file error: {e}")
        flash(f'Error deleting file: {str(e)}', 'error')
    
    return redirect(url_for('data_management'))

@app.route('/api/training_status')
def get_training_status():
    """API endpoint to get current training status"""
    return jsonify(training_state)

@app.route('/process_uploaded', methods=['POST'])
def process_uploaded():
    """Process all uploaded images"""
    try:
        processed_count = 0
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath) and image_processor.validate_image(filepath):
                processed_path = os.path.join(PROCESSED_FOLDER, filename)
                if image_processor.process_image(filepath, processed_path):
                    processed_count += 1
        
        flash(f'Processed {processed_count} images', 'success')
    
    except Exception as e:
        logging.error(f"Process uploaded error: {e}")
        flash(f'Error processing images: {str(e)}', 'error')
    
    return redirect(url_for('training'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
