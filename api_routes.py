import os
import json
import time
from pathlib import Path
from flask import jsonify, request, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from datetime import datetime
import zipfile
import io
from utils.tree_detector import TreeDetector
from utils.dataset_manager import DatasetManager
import zipfile
from pathlib import Path

def create_api_routes(app, enhanced_trainer, file_handler, image_processor, dataset_manager):
    """Create comprehensive API routes for the YOLOv8 dashboard"""
    
    @app.route('/api/training_status')
    def api_training_status():
        """Get comprehensive training status"""
        try:
            status = enhanced_trainer.get_training_status()
            return jsonify(status)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/start_training', methods=['POST'])
    def api_start_training():
        """Start enhanced training with comprehensive features"""
        try:
            data = request.get_json()
            dataset_path = data.get('dataset_path', 'datasets')
            resume = data.get('resume', False)
            mode = data.get('mode', 'new')
            
            # Update configuration if provided
            if 'config' in data:
                enhanced_trainer.config.update(data['config'])
                enhanced_trainer.save_config()
            
            success, message = enhanced_trainer.start_training(dataset_path, resume, mode)
            
            return jsonify({
                'success': success,
                'message': message
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/stop_training', methods=['POST'])
    def api_stop_training():
        """Stop training gracefully"""
        try:
            success, message = enhanced_trainer.stop_training()
            return jsonify({
                'success': success,
                'message': message
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/training_metrics')
    def api_training_metrics():
        """Get detailed training metrics"""
        try:
            status = enhanced_trainer.get_training_status()
            return jsonify({
                'metrics': status['metrics'],
                'current_epoch': status['current_epoch'],
                'total_epochs': status['total_epochs'],
                'progress_percentage': status['progress_percentage'],
                'processing_speed': status['processing_speed'],
                'gpu_memory_usage': status['gpu_memory_usage'],
                'failed_images_count': status['failed_images_count']
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/confusion_matrix/<int:epoch>')
    def api_confusion_matrix(epoch):
        """Get confusion matrix for specific epoch"""
        try:
            matrix_path = f'confusion_matrices/confusion_matrix_epoch_{epoch}.png'
            if os.path.exists(matrix_path):
                return send_file(matrix_path)
            else:
                return jsonify({'error': 'Confusion matrix not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/prediction_samples/<int:epoch>')
    def api_prediction_samples(epoch):
        """Get prediction samples for specific epoch"""
        try:
            samples_path = f'predictions/sample_predictions_epoch_{epoch}.png'
            if os.path.exists(samples_path):
                return send_file(samples_path)
            else:
                return jsonify({'error': 'Prediction samples not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/export_annotations', methods=['POST'])
    def api_export_annotations():
        """Export annotations in YOLO format"""
        try:
            data = request.get_json()
            annotations = data.get('annotations', {})
            classes = data.get('classes', [])
            images = data.get('images', [])
            
            # Create YOLO format files
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Create classes.names file
                zip_file.writestr('classes.names', '\n'.join(classes))
                
                # Create data.yaml
                yaml_content = f"""
path: ./dataset
train: train/images
val: val/images
test: test/images

nc: {len(classes)}
names: {classes}
"""
                zip_file.writestr('data.yaml', yaml_content)
                
                # Process annotations
                for image_name, image_annotations in annotations.items():
                    if image_annotations:
                        # Convert annotations to YOLO format
                        yolo_annotations = []
                        for ann in image_annotations:
                            # Convert to YOLO format (normalized coordinates)
                            x_center = (ann['x'] + ann['width'] / 2) / 640  # Assuming 640px image
                            y_center = (ann['y'] + ann['height'] / 2) / 640
                            width = ann['width'] / 640
                            height = ann['height'] / 640
                            
                            yolo_annotations.append(f"{ann['class']} {x_center} {y_center} {width} {height}")
                        
                        # Save annotation file
                        txt_filename = os.path.splitext(image_name)[0] + '.txt'
                        zip_file.writestr(f'labels/{txt_filename}', '\n'.join(yolo_annotations))
            
            zip_buffer.seek(0)
            
            return send_file(
                io.BytesIO(zip_buffer.read()),
                mimetype='application/zip',
                as_attachment=True,
                download_name='yolo_annotations.zip'
            )
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/inference', methods=['POST'])
    def api_inference():
        """Run inference on uploaded images"""
        try:
            # Get form data
            model_name = request.form.get('model')
            confidence = float(request.form.get('confidence', 0.5))
            iou = float(request.form.get('iou', 0.45))
            
            if not model_name:
                return jsonify({'error': 'No model selected'}), 400
            
            # Check if model exists
            model_path = f'models/{model_name}'
            if not os.path.exists(model_path):
                return jsonify({'error': 'Model not found'}), 404
            
            # Process uploaded images
            results = []
            
            for file in request.files.getlist('images'):
                if file and file.filename:
                    # Save uploaded image temporarily
                    filename = secure_filename(file.filename)
                    temp_path = f'temp_{filename}'
                    file.save(temp_path)
                    
                    try:
                        # Run inference (simulated for now)
                        detections = simulate_inference(temp_path, confidence, iou)
                        
                        # Process image for display
                        processed_image_url = process_inference_image(temp_path, detections)
                        
                        results.append({
                            'filename': filename,
                            'image_url': processed_image_url,
                            'detections': detections,
                            'processing_time': np.random.uniform(50, 200)  # Simulated
                        })
                        
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/upload_dataset', methods=['POST'])
    def api_upload_dataset():
        """Upload and organize dataset for training"""
        try:
            uploaded_files = []
            
            for file in request.files.getlist('files'):
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    
                    # Handle ZIP files
                    if filename.lower().endswith('.zip'):
                        # Extract ZIP file
                        temp_zip_path = f'temp_{filename}'
                        file.save(temp_zip_path)
                        
                        try:
                            extracted_files = file_handler.extract_zip(temp_zip_path, 'uploads')
                            uploaded_files.extend(extracted_files)
                        finally:
                            if os.path.exists(temp_zip_path):
                                os.remove(temp_zip_path)
                    
                    # Handle image files
                    elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                        file_path = os.path.join('uploads', filename)
                        file.save(file_path)
                        
                        # Validate and process image
                        if image_processor.validate_image(file_path):
                            processed_path = image_processor.process_image(file_path, 'processed')
                            uploaded_files.append({
                                'original': filename,
                                'processed': os.path.basename(processed_path),
                                'status': 'success'
                            })
                        else:
                            uploaded_files.append({
                                'original': filename,
                                'status': 'failed',
                                'error': 'Invalid image format'
                            })
            
            return jsonify({
                'success': True,
                'uploaded_files': uploaded_files,
                'total_files': len(uploaded_files)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/augment_data', methods=['POST'])
    def api_augment_data():
        """Apply data augmentation to training dataset"""
        try:
            data = request.get_json()
            augmentation_params = data.get('augmentation', {})
            
            # Apply augmentation (simulated)
            augmented_count = apply_data_augmentation(augmentation_params)
            
            return jsonify({
                'success': True,
                'augmented_images': augmented_count,
                'message': f'Generated {augmented_count} augmented images'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/validate_model', methods=['POST'])
    def api_validate_model():
        """Validate trained model performance"""
        try:
            data = request.get_json()
            model_name = data.get('model')
            dataset_path = data.get('dataset', 'datasets')
            
            if not model_name:
                return jsonify({'error': 'No model specified'}), 400
            
            # Run model validation (simulated)
            validation_results = simulate_model_validation()
            
            return jsonify({
                'success': True,
                'validation_results': validation_results
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/download_results')
    def api_download_results():
        """Download training results and reports"""
        try:
            # Create comprehensive results ZIP
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add training reports
                results_dir = Path('results')
                if results_dir.exists():
                    for file_path in results_dir.glob('*'):
                        if file_path.is_file():
                            zip_file.write(file_path, f'reports/{file_path.name}')
                
                # Add confusion matrices
                cm_dir = Path('confusion_matrices')
                if cm_dir.exists():
                    for file_path in cm_dir.glob('*.png'):
                        zip_file.write(file_path, f'confusion_matrices/{file_path.name}')
                
                # Add prediction samples
                pred_dir = Path('predictions')
                if pred_dir.exists():
                    for file_path in pred_dir.glob('*.png'):
                        zip_file.write(file_path, f'predictions/{file_path.name}')
                
                # Add trained models
                models_dir = Path('models')
                if models_dir.exists():
                    for file_path in models_dir.glob('*.pt'):
                        zip_file.write(file_path, f'models/{file_path.name}')
            
            zip_buffer.seek(0)
            
            return send_file(
                io.BytesIO(zip_buffer.read()),
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'yolo_training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
            )
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/auto_detect_trees', methods=['POST'])
    def api_auto_detect_trees():
        """Automatically detect palm trees in uploaded images and save them to the dataset."""
        try:
            files = request.files.getlist('images')
            if not files or all(f.filename == '' for f in files):
                return jsonify({'error': 'No images selected'}), 400

            tree_detector = TreeDetector()
            images_dir = Path('datasets/train/images')
            labels_dir = Path('datasets/train/labels')
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            processed_files = 0
            total_detections = 0

            for file in files:
                filename = secure_filename(file.filename)
                if filename.lower().endswith('.zip'):
                    # Handle zip files
                    temp_zip_path = f"temp_{filename}"
                    file.save(temp_zip_path)
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            if member.lower().endswith(('.png', '.jpg', '.jpeg')):
                                with zip_ref.open(member) as image_file:
                                    image_filename = secure_filename(Path(member).name)
                                    image_path = images_dir / image_filename
                                    with open(image_path, 'wb') as f:
                                        f.write(image_file.read())

                                    bounding_boxes = tree_detector.detect_trees(str(image_path))
                                    if bounding_boxes:
                                        label_path = labels_dir / f"{image_path.stem}.txt"
                                        tree_detector._save_yolo_annotations(str(label_path), bounding_boxes)
                                        total_detections += len(bounding_boxes)
                                    processed_files += 1
                    os.remove(temp_zip_path)
                elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Handle single image files
                    image_path = images_dir / filename
                    file.save(image_path)

                    bounding_boxes = tree_detector.detect_trees(str(image_path))
                    if bounding_boxes:
                        label_path = labels_dir / f"{image_path.stem}.txt"
                        tree_detector._save_yolo_annotations(str(label_path), bounding_boxes)
                        total_detections += len(bounding_boxes)
                    processed_files += 1

            return jsonify({
                'success': True,
                'message': f'Processed {processed_files} images, found {total_detections} trees.'
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/download_auto_detection_results')
    def api_download_auto_detection_results():
        """Download auto detection results from augmented_data folder"""
        try:
            # Create ZIP file with all results
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                augmented_data_dir = Path('augmented_data')
                
                if augmented_data_dir.exists():
                    for file_path in augmented_data_dir.glob('*'):
                        if file_path.is_file():
                            # Add file to ZIP with appropriate folder structure
                            if file_path.suffix == '.txt':
                                zip_file.write(file_path, f'labels/{file_path.name}')
                            else:
                                zip_file.write(file_path, f'images/{file_path.name}')
                
                # Add classes.txt if it exists
                classes_path = augmented_data_dir / 'classes.txt'
                if classes_path.exists():
                    zip_file.write(classes_path, 'classes.txt')
            
            zip_buffer.seek(0)
            
            return send_file(
                io.BytesIO(zip_buffer.read()),
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'palm_tree_detection_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
            )
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500


def simulate_inference(image_path, confidence, iou):
    """Simulate inference results"""
    try:
        # Load image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        h, w, _ = img.shape
        
        # Generate random detections
        detections = []
        num_detections = np.random.randint(1, 6)
        
        classes = ['person', 'car', 'truck', 'bike', 'motorcycle']
        
        for _ in range(num_detections):
            # Random bounding box
            x1 = np.random.randint(0, w//2)
            y1 = np.random.randint(0, h//2)
            x2 = np.random.randint(x1 + 50, w)
            y2 = np.random.randint(y1 + 50, h)
            
            # Random confidence above threshold
            conf = np.random.uniform(confidence, 1.0)
            
            # Random class
            class_idx = np.random.randint(0, len(classes))
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class': class_idx,
                'class_name': classes[class_idx]
            })
        
        return detections
        
    except Exception as e:
        print(f"Error in simulate_inference: {e}")
        return []


def process_inference_image(image_path, detections):
    """Process image with detection overlays"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save processed image
        output_filename = f"inference_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        output_path = f"static/inference_results/{output_filename}"
        
        # Ensure output directory exists
        os.makedirs("static/inference_results", exist_ok=True)
        
        cv2.imwrite(output_path, img)
        
        return f"/static/inference_results/{output_filename}"
        
    except Exception as e:
        print(f"Error processing inference image: {e}")
        return None


def apply_data_augmentation(params):
    """Apply data augmentation techniques"""
    try:
        # Simulate augmentation process
        base_count = len(os.listdir('uploads')) if os.path.exists('uploads') else 0
        augmentation_factor = params.get('factor', 2)
        
        # Simulate generating augmented images
        augmented_count = base_count * augmentation_factor
        
        return augmented_count
        
    except Exception as e:
        print(f"Error in data augmentation: {e}")
        return 0


def simulate_model_validation():
    """Simulate model validation results"""
    return {
        'accuracy': np.random.uniform(0.8, 0.95),
        'precision': np.random.uniform(0.75, 0.9),
        'recall': np.random.uniform(0.7, 0.88),
        'f1_score': np.random.uniform(0.72, 0.89),
        'map50': np.random.uniform(0.65, 0.85),
        'map50_95': np.random.uniform(0.45, 0.65),
        'inference_time': np.random.uniform(15, 45),  # ms
        'model_size': np.random.uniform(20, 50)  # MB
    }

    # Dataset Management API Routes
    @app.route('/api/dataset/info')
    def api_dataset_info():
        """Get dataset information"""
        try:
            info = dataset_manager.get_dataset_info()
            return jsonify(info)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dataset/validate')
    def api_dataset_validate():
        """Validate dataset integrity"""
        try:
            validation_result = dataset_manager.validate_dataset()
            return jsonify(validation_result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dataset/cleanup', methods=['POST'])
    def api_dataset_cleanup():
        """Clean up orphaned files"""
        try:
            cleanup_result = dataset_manager.cleanup_orphaned_files()
            return jsonify(cleanup_result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/annotations/save', methods=['POST'])
    def api_save_annotations():
        """Save annotations to dataset"""
        try:
            data = request.get_json()
            image_path = data.get('image_path')
            annotations = data.get('annotations', [])
            output_name = data.get('output_name')
            
            if not image_path or not os.path.exists(image_path):
                return jsonify({'error': 'Image not found'}), 404
            
            # Convert annotations to the format expected by dataset manager
            processed_annotations = []
            for ann in annotations:
                processed_annotations.append({
                    'class': ann.get('class', 'unknown'),
                    'x': ann.get('x', 0),
                    'y': ann.get('y', 0),
                    'width': ann.get('width', 0),
                    'height': ann.get('height', 0)
                })
            
            result = dataset_manager.save_annotated_image(
                image_path, 
                processed_annotations, 
                output_name
            )
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/annotations/save_batch', methods=['POST'])
    def api_save_annotations_batch():
        """Save multiple annotations in batch"""
        try:
            data = request.get_json()
            image_annotations = data.get('image_annotations', [])
            
            if not image_annotations:
                return jsonify({'error': 'No annotations provided'}), 400
            
            # Process each annotation
            processed_batch = []
            for item in image_annotations:
                if os.path.exists(item.get('image_path', '')):
                    processed_batch.append({
                        'image_path': item['image_path'],
                        'annotations': item.get('annotations', []),
                        'output_name': item.get('output_name')
                    })
            
            result = dataset_manager.save_annotated_images_batch(processed_batch)
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/annotations/auto_detect', methods=['POST'])
    def api_auto_detect_annotations():
        """Auto-detect annotations using tree detector"""
        try:
            data = request.get_json()
            image_paths = data.get('image_paths', [])
            
            if not image_paths:
                return jsonify({'error': 'No images provided'}), 400
            
            results = []
            tree_detector = TreeDetector()
            
            for image_path in image_paths:
                if os.path.exists(image_path):
                    # Detect trees in the image
                    detections = tree_detector.detect_trees(image_path)
                    
                    # Convert detections to annotation format
                    annotations = []
                    for detection in detections:
                        # Assuming detection format from tree_detector
                        # You may need to adjust this based on actual output format
                        annotations.append({
                            'class': 'palm_tree',  # Default class for tree detection
                            'x': detection.get('x', 0),
                            'y': detection.get('y', 0),
                            'width': detection.get('width', 0),
                            'height': detection.get('height', 0)
                        })
                    
                    results.append({
                        'image_path': image_path,
                        'annotations': annotations,
                        'detection_count': len(annotations)
                    })
            
            return jsonify({
                'success': True,
                'results': results,
                'total_images': len(image_paths)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

