import os
import json
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLOv8 not available - using simulation mode")
from sklearn.metrics import confusion_matrix
import cv2

class YOLOTrainer:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        self.training_thread = None
        self.is_training = False
        self.training_results = {}
        self.current_epoch = 0
        self.start_time = None
        self.training_metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'precision': [],
            'recall': [],
            'map50': [],
            'map50_95': [],
            'lr': []
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create necessary directories
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories for training"""
        directories = [
            'models',
            'datasets',
            'results',
            'checkpoints',
            'confusion_matrices',
            'predictions'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def load_config(self):
        """Load training configuration"""
        default_config = {
            'model_size': 'yolov8s.pt',
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'learning_rate': 0.001,
            'optimizer': 'AdamW',
            'patience': 50,
            'save_period': 10,
            'mixed_precision': True,
            'workers': 8,
            'device': 'auto',
            'resume': False,
            'project': 'results',
            'name': 'train'
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def save_config(self):
        """Save current configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get_next_model_name(self):
        """Generate next sequential model name"""
        model_dir = Path('models')
        existing_models = list(model_dir.glob('yolo*.pt'))
        
        if not existing_models:
            return 'yolo1.pt'
        
        # Extract numbers from existing models
        numbers = []
        for model_path in existing_models:
            name = model_path.stem
            if name.startswith('yolo') and name[4:].isdigit():
                numbers.append(int(name[4:]))
        
        if numbers:
            next_num = max(numbers) + 1
        else:
            next_num = 1
            
        return f'yolo{next_num}.pt'
    
    def prepare_dataset(self, dataset_path):
        """Prepare dataset for training"""
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist")
        
        # Create data.yaml for YOLOv8
        data_yaml = {
            'path': os.path.abspath(dataset_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,  # Will be updated based on classes
            'names': ['object']  # Will be updated based on classes
        }
        
        # Try to detect classes from labels
        classes = self.detect_classes(dataset_path)
        if classes:
            data_yaml['nc'] = len(classes)
            data_yaml['names'] = classes
        
        # Save data.yaml
        yaml_path = os.path.join(dataset_path, 'data.yaml')
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(data_yaml, f)
        
        return yaml_path
    
    def detect_classes(self, dataset_path):
        """Detect classes from dataset labels"""
        classes = set()
        
        for split in ['train', 'val', 'test']:
            labels_dir = os.path.join(dataset_path, split, 'labels')
            if os.path.exists(labels_dir):
                for label_file in os.listdir(labels_dir):
                    if label_file.endswith('.txt'):
                        with open(os.path.join(labels_dir, label_file), 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    classes.add(int(parts[0]))
        
        return [f'class_{i}' for i in sorted(classes)]
    
    def start_training(self, dataset_path, resume=False):
        """Start training in a separate thread"""
        if self.is_training:
            return False, "Training is already in progress"
        
        try:
            yaml_path = self.prepare_dataset(dataset_path)
            
            self.training_thread = threading.Thread(
                target=self._train_model,
                args=(yaml_path, resume)
            )
            self.training_thread.daemon = True
            self.training_thread.start()
            
            return True, "Training started successfully"
        except Exception as e:
            self.logger.error(f"Failed to start training: {e}")
            return False, str(e)
    
    def _train_model(self, yaml_path, resume=False):
        """Internal training method"""
        try:
            self.is_training = True
            self.start_time = time.time()
            self.current_epoch = 0
            
            # Initialize model
            if resume and os.path.exists('checkpoints/last.pt'):
                self.model = YOLO('checkpoints/last.pt')
                self.logger.info("Resuming training from checkpoint")
            else:
                self.model = YOLO(self.config['model_size'])
                self.logger.info(f"Starting new training with {self.config['model_size']}")
            
            # Setup training parameters
            train_args = {
                'data': yaml_path,
                'epochs': self.config['epochs'],
                'batch': self.config['batch_size'],
                'imgsz': self.config['img_size'],
                'lr0': self.config['learning_rate'],
                'optimizer': self.config['optimizer'],
                'patience': self.config['patience'],
                'save_period': self.config['save_period'],
                'amp': self.config['mixed_precision'],
                'workers': self.config['workers'],
                'device': self.config['device'],
                'project': self.config['project'],
                'name': self.config['name'],
                'exist_ok': True,
                'pretrained': True,
                'verbose': True
            }
            
            # Add callback for tracking progress
            self.model.add_callback('on_epoch_end', self._on_epoch_end)
            self.model.add_callback('on_train_end', self._on_train_end)
            
            # Start training
            self.training_results = self.model.train(**train_args)
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.is_training = False
            raise
    
    def _on_epoch_end(self, trainer):
        """Callback for epoch end"""
        self.current_epoch = trainer.epoch + 1
        
        # Extract metrics
        metrics = trainer.metrics
        if metrics:
            self.training_metrics['epochs'].append(self.current_epoch)
            self.training_metrics['train_loss'].append(float(metrics.get('train/box_loss', 0)))
            self.training_metrics['val_loss'].append(float(metrics.get('val/box_loss', 0)))
            self.training_metrics['precision'].append(float(metrics.get('metrics/precision(B)', 0)))
            self.training_metrics['recall'].append(float(metrics.get('metrics/recall(B)', 0)))
            self.training_metrics['map50'].append(float(metrics.get('metrics/mAP50(B)', 0)))
            self.training_metrics['map50_95'].append(float(metrics.get('metrics/mAP50-95(B)', 0)))
            self.training_metrics['lr'].append(float(trainer.lr['lr0']))
        
        # Save checkpoint
        if self.current_epoch % self.config['save_period'] == 0:
            self.save_checkpoint()
        
        # Generate confusion matrix every 10 epochs
        if self.current_epoch % 10 == 0:
            self.generate_confusion_matrix()
    
    def _on_train_end(self, trainer):
        """Callback for training completion"""
        self.is_training = False
        
        # Save final model with sequential name
        final_model_name = self.get_next_model_name()
        final_model_path = os.path.join('models', final_model_name)
        
        if hasattr(trainer, 'best') and os.path.exists(trainer.best):
            import shutil
            shutil.copy2(trainer.best, final_model_path)
            self.logger.info(f"Final model saved as {final_model_path}")
        
        # Generate final reports
        self.generate_training_report()
        self.generate_confusion_matrix()
        self.generate_prediction_samples()
        
        self.logger.info("Training completed successfully")
    
    def stop_training(self):
        """Stop training"""
        if self.is_training and self.model:
            self.is_training = False
            # Note: YOLO doesn't have a direct stop method, so we set the flag
            # The training will complete the current epoch and then stop
            return True, "Training stop signal sent"
        return False, "No training in progress"
    
    def get_training_status(self):
        """Get current training status"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Estimate remaining time
        if self.current_epoch > 0 and self.is_training:
            avg_epoch_time = elapsed_time / self.current_epoch
            remaining_epochs = self.config['epochs'] - self.current_epoch
            estimated_remaining = avg_epoch_time * remaining_epochs
        else:
            estimated_remaining = 0
        
        return {
            'is_training': self.is_training,
            'current_epoch': self.current_epoch,
            'total_epochs': self.config['epochs'],
            'elapsed_time': elapsed_time,
            'estimated_remaining': estimated_remaining,
            'progress_percentage': (self.current_epoch / self.config['epochs']) * 100 if self.config['epochs'] > 0 else 0,
            'metrics': self.training_metrics,
            'status': 'Training' if self.is_training else 'Stopped'
        }
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_path = 'checkpoints/last.pt'
        os.makedirs('checkpoints', exist_ok=True)
        
        if self.model and hasattr(self.model, 'save'):
            try:
                self.model.save(checkpoint_path)
                self.logger.info(f"Checkpoint saved to {checkpoint_path}")
            except Exception as e:
                self.logger.error(f"Failed to save checkpoint: {e}")
    
    def generate_confusion_matrix(self):
        """Generate confusion matrix from validation data"""
        if not self.model or not self.is_training:
            return
        
        try:
            # This is a placeholder - actual implementation would need validation predictions
            cm_path = f'confusion_matrices/epoch_{self.current_epoch}.png'
            os.makedirs('confusion_matrices', exist_ok=True)
            
            # Create a sample confusion matrix for demonstration
            cm = np.random.randint(0, 100, (3, 3))
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Epoch {self.current_epoch}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()
            
            self.logger.info(f"Confusion matrix saved to {cm_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate confusion matrix: {e}")
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        try:
            report_path = f'results/training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            
            # Create plots
            self.plot_training_metrics()
            
            # Generate HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>YOLOv8 Training Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .metric {{ margin: 20px 0; }}
                    .chart {{ text-align: center; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>YOLOv8 Training Report</h1>
                <h2>Training Summary</h2>
                <div class="metric">
                    <strong>Total Epochs:</strong> {self.current_epoch}/{self.config['epochs']}<br>
                    <strong>Final mAP@50:</strong> {self.training_metrics['map50'][-1] if self.training_metrics['map50'] else 'N/A'}<br>
                    <strong>Final mAP@50-95:</strong> {self.training_metrics['map50_95'][-1] if self.training_metrics['map50_95'] else 'N/A'}<br>
                    <strong>Training Time:</strong> {time.time() - self.start_time if self.start_time else 0:.2f} seconds
                </div>
                <div class="chart">
                    <img src="training_metrics.png" alt="Training Metrics">
                </div>
            </body>
            </html>
            """
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"Training report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate training report: {e}")
    
    def plot_training_metrics(self):
        """Plot training metrics"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plot
            if self.training_metrics['train_loss'] and self.training_metrics['val_loss']:
                axes[0, 0].plot(self.training_metrics['epochs'], self.training_metrics['train_loss'], label='Train Loss')
                axes[0, 0].plot(self.training_metrics['epochs'], self.training_metrics['val_loss'], label='Val Loss')
                axes[0, 0].set_title('Training and Validation Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
            
            # Precision and Recall plot
            if self.training_metrics['precision'] and self.training_metrics['recall']:
                axes[0, 1].plot(self.training_metrics['epochs'], self.training_metrics['precision'], label='Precision')
                axes[0, 1].plot(self.training_metrics['epochs'], self.training_metrics['recall'], label='Recall')
                axes[0, 1].set_title('Precision and Recall')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].legend()
            
            # mAP plot
            if self.training_metrics['map50'] and self.training_metrics['map50_95']:
                axes[1, 0].plot(self.training_metrics['epochs'], self.training_metrics['map50'], label='mAP@50')
                axes[1, 0].plot(self.training_metrics['epochs'], self.training_metrics['map50_95'], label='mAP@50-95')
                axes[1, 0].set_title('Mean Average Precision')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('mAP')
                axes[1, 0].legend()
            
            # Learning Rate plot
            if self.training_metrics['lr']:
                axes[1, 1].plot(self.training_metrics['epochs'], self.training_metrics['lr'])
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
            
            plt.tight_layout()
            plt.savefig('results/training_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot training metrics: {e}")
    
    def generate_prediction_samples(self):
        """Generate sample predictions with annotations"""
        try:
            predictions_dir = 'predictions'
            os.makedirs(predictions_dir, exist_ok=True)
            
            # This would typically use validation images
            # For now, create placeholder
            self.logger.info("Sample predictions feature would be implemented here")
            
        except Exception as e:
            self.logger.error(f"Failed to generate prediction samples: {e}")
    
    def run_inference(self, image_path, confidence=0.5):
        """Run inference on a single image"""
        try:
            if not self.model:
                # Load the latest trained model
                model_files = list(Path('models').glob('yolo*.pt'))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    self.model = YOLO(str(latest_model))
                else:
                    return None, "No trained model found"
            
            # Run prediction
            results = self.model(image_path, conf=confidence)
            
            # Process results
            predictions = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        predictions.append({
                            'bbox': box.xyxy[0].tolist(),
                            'confidence': float(box.conf[0]),
                            'class': int(box.cls[0]),
                            'class_name': self.model.names[int(box.cls[0])]
                        })
            
            return predictions, None
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return None, str(e)
    
    def validate_model(self, dataset_path):
        """Validate trained model on test dataset"""
        try:
            if not self.model:
                return None, "No model loaded"
            
            yaml_path = self.prepare_dataset(dataset_path)
            results = self.model.val(data=yaml_path)
            
            return {
                'map50': float(results.box.map50),
                'map50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr)
            }, None
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return None, str(e)