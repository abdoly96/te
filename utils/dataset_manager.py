import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
import yaml

class DatasetManager:
    """Manages dataset structure and YOLO format files"""
    
    def __init__(self, base_path='datasets'):
        self.base_path = Path(base_path)
        self.train_images_path = self.base_path / 'train' / 'images'
        self.train_labels_path = self.base_path / 'train' / 'labels'
        self.val_images_path = self.base_path / 'val' / 'images'
        self.val_labels_path = self.base_path / 'val' / 'labels'
        self.data_yaml_path = self.base_path / 'data.yaml'
        
        # Create directory structure
        self._create_directory_structure()
        
        # Initialize classes
        self.classes = []
        self.class_mapping = {}
        
        # Load existing data.yaml if exists
        self._load_data_yaml()
        
        self.logger = logging.getLogger(__name__)
    
    def _create_directory_structure(self):
        """Create the required directory structure"""
        directories = [
            self.train_images_path,
            self.train_labels_path,
            self.val_images_path,
            self.val_labels_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_data_yaml(self):
        """Load existing data.yaml file"""
        if self.data_yaml_path.exists():
            try:
                with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    self.classes = data.get('names', [])
                    # Create class mapping
                    self.class_mapping = {name: idx for idx, name in enumerate(self.classes)}
            except Exception as e:
                self.logger.error(f"Error loading data.yaml: {e}")
                self.classes = []
                self.class_mapping = {}
    
    def _save_data_yaml(self):
        """Save data.yaml file with current configuration"""
        try:
            data = {
                'path': str(self.base_path.absolute()),
                'train': str(self.train_images_path.relative_to(self.base_path)),
                'val': str(self.val_images_path.relative_to(self.base_path)),
                'nc': len(self.classes),
                'names': self.classes
            }
            
            with open(self.data_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            self.logger.info(f"data.yaml saved successfully with {len(self.classes)} classes")
            return True
        except Exception as e:
            self.logger.error(f"Error saving data.yaml: {e}")
            return False
    
    def add_class(self, class_name):
        """Add a new class if it doesn't exist"""
        if class_name not in self.classes:
            self.classes.append(class_name)
            self.class_mapping[class_name] = len(self.classes) - 1
            self._save_data_yaml()
            self.logger.info(f"Added new class: {class_name}")
            return True
        return False
    
    def get_class_id(self, class_name):
        """Get class ID for a given class name"""
        if class_name not in self.class_mapping:
            self.add_class(class_name)
        return self.class_mapping[class_name]
    
    def save_annotated_image(self, image_path, annotations, output_name=None):
        """
        Save annotated image and its YOLO label file
        
        Args:
            image_path: Path to source image
            annotations: List of annotation objects with format:
                [{'class': 'class_name', 'x': x_center, 'y': y_center, 'width': width, 'height': height}]
            output_name: Optional custom name for the files
        
        Returns:
            dict: {'success': bool, 'image_path': str, 'label_path': str, 'message': str}
        """
        try:
            # Generate output name if not provided
            if output_name is None:
                output_name = Path(image_path).stem
            
            # Check if image has annotations
            if not annotations:
                return {
                    'success': False,
                    'message': 'No annotations found, skipping image'
                }
            
            # Copy image to train/images
            image_ext = Path(image_path).suffix
            target_image_path = self.train_images_path / f"{output_name}{image_ext}"
            
            # Ensure unique filename
            counter = 1
            original_name = output_name
            while target_image_path.exists():
                output_name = f"{original_name}_{counter}"
                target_image_path = self.train_images_path / f"{output_name}{image_ext}"
                counter += 1
            
            # Copy image
            shutil.copy2(image_path, target_image_path)
            
            # Create YOLO label file
            label_path = self.train_labels_path / f"{output_name}.txt"
            
            with open(label_path, 'w') as f:
                for annotation in annotations:
                    class_id = self.get_class_id(annotation['class'])
                    # Convert to YOLO format (normalized coordinates)
                    x_center = annotation['x'] / 100.0  # Assuming coordinates are in percentage
                    y_center = annotation['y'] / 100.0
                    width = annotation['width'] / 100.0
                    height = annotation['height'] / 100.0
                    
                    # Ensure coordinates are within [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            self.logger.info(f"Saved annotated image: {target_image_path} with {len(annotations)} annotations")
            
            return {
                'success': True,
                'image_path': str(target_image_path),
                'label_path': str(label_path),
                'message': f"Successfully saved {len(annotations)} annotations"
            }
            
        except Exception as e:
            self.logger.error(f"Error saving annotated image: {e}")
            return {
                'success': False,
                'message': f"Error: {str(e)}"
            }
    
    def save_annotated_images_batch(self, image_annotations):
        """
        Save multiple annotated images in batch
        
        Args:
            image_annotations: List of dicts with format:
                [{'image_path': 'path/to/image.jpg', 'annotations': [...], 'output_name': 'optional_name'}]
        
        Returns:
            dict: Summary of the operation
        """
        results = []
        successful = 0
        failed = 0
        
        for item in image_annotations:
            result = self.save_annotated_image(
                item['image_path'],
                item['annotations'],
                item.get('output_name')
            )
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        # Update data.yaml
        self._save_data_yaml()
        
        return {
            'total': len(image_annotations),
            'successful': successful,
            'failed': failed,
            'results': results
        }
    
    def get_dataset_info(self):
        """Get information about the current dataset"""
        try:
            train_images = list(self.train_images_path.glob('*'))
            train_labels = list(self.train_labels_path.glob('*.txt'))
            val_images = list(self.val_images_path.glob('*'))
            val_labels = list(self.val_labels_path.glob('*.txt'))
            
            # Count images by extension
            image_extensions = {}
            for img_path in train_images + val_images:
                ext = img_path.suffix.lower()
                image_extensions[ext] = image_extensions.get(ext, 0) + 1
            
            return {
                'train_images': len(train_images),
                'train_labels': len(train_labels),
                'val_images': len(val_images),
                'val_labels': len(val_labels),
                'classes': self.classes,
                'class_count': len(self.classes),
                'image_extensions': image_extensions,
                'data_yaml_path': str(self.data_yaml_path),
                'base_path': str(self.base_path.absolute())
            }
        except Exception as e:
            self.logger.error(f"Error getting dataset info: {e}")
            return {}
    
    def validate_dataset(self):
        """Validate dataset integrity"""
        issues = []
        
        try:
            # Check if train images have corresponding labels
            train_images = list(self.train_images_path.glob('*'))
            for img_path in train_images:
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    label_path = self.train_labels_path / f"{img_path.stem}.txt"
                    if not label_path.exists():
                        issues.append(f"Missing label for {img_path.name}")
            
            # Check if labels have corresponding images
            train_labels = list(self.train_labels_path.glob('*.txt'))
            for label_path in train_labels:
                # Check for multiple possible image extensions
                found = False
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_path = self.train_images_path / f"{label_path.stem}{ext}"
                    if img_path.exists():
                        found = True
                        break
                
                if not found:
                    issues.append(f"Missing image for {label_path.name}")
            
            # Validate YOLO format in label files
            for label_path in train_labels:
                try:
                    with open(label_path, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) != 5:
                                    issues.append(f"Invalid format in {label_path.name}:{line_num}")
                                else:
                                    try:
                                        class_id = int(parts[0])
                                        coords = [float(x) for x in parts[1:]]
                                        if class_id < 0 or class_id >= len(self.classes):
                                            issues.append(f"Invalid class ID {class_id} in {label_path.name}:{line_num}")
                                        if any(coord < 0 or coord > 1 for coord in coords):
                                            issues.append(f"Invalid coordinates in {label_path.name}:{line_num}")
                                    except ValueError:
                                        issues.append(f"Invalid values in {label_path.name}:{line_num}")
                except Exception as e:
                    issues.append(f"Error reading {label_path.name}: {e}")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'issue_count': len(issues)
            }
            
        except Exception as e:
            self.logger.error(f"Error validating dataset: {e}")
            return {
                'valid': False,
                'issues': [f"Validation error: {e}"],
                'issue_count': 1
            }
    
    def cleanup_orphaned_files(self):
        """Remove orphaned files (images without labels or labels without images)"""
        try:
            removed_count = 0
            
            # Remove images without labels
            train_images = list(self.train_images_path.glob('*'))
            for img_path in train_images:
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    label_path = self.train_labels_path / f"{img_path.stem}.txt"
                    if not label_path.exists():
                        img_path.unlink()
                        removed_count += 1
                        self.logger.info(f"Removed orphaned image: {img_path.name}")
            
            # Remove labels without images
            train_labels = list(self.train_labels_path.glob('*.txt'))
            for label_path in train_labels:
                found = False
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_path = self.train_images_path / f"{label_path.stem}{ext}"
                    if img_path.exists():
                        found = True
                        break
                
                if not found:
                    label_path.unlink()
                    removed_count += 1
                    self.logger.info(f"Removed orphaned label: {label_path.name}")
            
            return {
                'success': True,
                'removed_count': removed_count,
                'message': f"Cleaned up {removed_count} orphaned files"
            }
            
        except Exception as e:
            self.logger.error(f"Error cleaning up orphaned files: {e}")
            return {
                'success': False,
                'message': f"Error: {str(e)}"
            }
