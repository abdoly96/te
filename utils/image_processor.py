import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging

class ImageProcessor:
    """Handle image processing tasks for YOLOv8 training"""
    
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
        self.target_size = (640, 640)  # Default YOLO input size
        
    def validate_image(self, image_path):
        """Validate if the file is a valid image"""
        try:
            # Check file extension
            _, ext = os.path.splitext(image_path)
            if ext.lower() not in self.supported_formats:
                logging.warning(f"Unsupported format: {ext}")
                return False
            
            # Try to open and verify the image
            with Image.open(image_path) as img:
                img.verify()
            
            # Check if image can be loaded with OpenCV
            cv_img = cv2.imread(image_path)
            if cv_img is None:
                logging.warning(f"OpenCV cannot read image: {image_path}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Image validation failed for {image_path}: {e}")
            return False
    
    def process_image(self, input_path, output_path, target_size=None):
        """Process a single image for training"""
        try:
            if target_size is None:
                target_size = self.target_size
            
            # Load image with PIL for better format support
            with Image.open(input_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image while maintaining aspect ratio
                img = self._resize_with_padding(img, target_size)
                
                # Apply image enhancements
                img = self._enhance_image(img)
                
                # Save processed image
                img.save(output_path, 'JPEG', quality=95)
                
                logging.info(f"Processed image: {input_path} -> {output_path}")
                return True
                
        except Exception as e:
            logging.error(f"Image processing failed for {input_path}: {e}")
            return False
    
    def _resize_with_padding(self, img, target_size):
        """Resize image while maintaining aspect ratio and adding padding"""
        original_width, original_height = img.size
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width / original_width, target_height / original_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_img = Image.new('RGB', target_size, (114, 114, 114))  # Gray padding
        
        # Calculate padding offsets
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # Paste resized image onto padded background
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
    
    def _enhance_image(self, img):
        """Apply image enhancements"""
        try:
            # Normalize brightness and contrast
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.05)
            
            # Slight sharpening
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.02)
            
            return img
            
        except Exception as e:
            logging.warning(f"Image enhancement failed: {e}")
            return img
    
    def denoise_image(self, img_array):
        """Apply denoising to image array"""
        try:
            # Apply Non-local Means Denoising
            if len(img_array.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
            
            return denoised
            
        except Exception as e:
            logging.warning(f"Denoising failed: {e}")
            return img_array
    
    def augment_image(self, img, augmentation_type='basic'):
        """Apply data augmentation to image"""
        try:
            if augmentation_type == 'basic':
                return self._basic_augmentation(img)
            elif augmentation_type == 'advanced':
                return self._advanced_augmentation(img)
            else:
                return img
                
        except Exception as e:
            logging.warning(f"Augmentation failed: {e}")
            return img
    
    def _basic_augmentation(self, img):
        """Apply basic augmentation transformations"""
        augmented_images = [img]  # Original image
        
        # Horizontal flip
        flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        augmented_images.append(flipped)
        
        # Rotation
        for angle in [-10, 10]:
            rotated = img.rotate(angle, expand=False, fillcolor=(114, 114, 114))
            augmented_images.append(rotated)
        
        # Brightness variations
        for factor in [0.8, 1.2]:
            enhancer = ImageEnhance.Brightness(img)
            bright_img = enhancer.enhance(factor)
            augmented_images.append(bright_img)
        
        return augmented_images
    
    def _advanced_augmentation(self, img):
        """Apply advanced augmentation transformations"""
        augmented_images = self._basic_augmentation(img)
        
        # Color variations
        for factor in [0.8, 1.2]:
            enhancer = ImageEnhance.Color(img)
            color_img = enhancer.enhance(factor)
            augmented_images.append(color_img)
        
        # Gaussian blur
        blurred = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        augmented_images.append(blurred)
        
        # Contrast variations
        for factor in [0.8, 1.2]:
            enhancer = ImageEnhance.Contrast(img)
            contrast_img = enhancer.enhance(factor)
            augmented_images.append(contrast_img)
        
        return augmented_images
    
    def batch_process_images(self, input_dir, output_dir, target_size=None):
        """Process all images in a directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        processed_count = 0
        error_count = 0
        
        for filename in os.listdir(input_dir):
            input_path = os.path.join(input_dir, filename)
            
            if os.path.isfile(input_path) and self.validate_image(input_path):
                # Generate output filename
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}.jpg"  # Convert all to JPEG
                output_path = os.path.join(output_dir, output_filename)
                
                if self.process_image(input_path, output_path, target_size):
                    processed_count += 1
                else:
                    error_count += 1
        
        logging.info(f"Batch processing complete: {processed_count} processed, {error_count} errors")
        return processed_count, error_count
    
    def get_image_info(self, image_path):
        """Get information about an image"""
        try:
            with Image.open(image_path) as img:
                return {
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'has_transparency': img.mode in ('RGBA', 'LA', 'P')
                }
        except Exception as e:
            logging.error(f"Failed to get image info for {image_path}: {e}")
            return None
    
    def create_thumbnail(self, image_path, thumbnail_path, size=(128, 128)):
        """Create a thumbnail of the image"""
        try:
            with Image.open(image_path) as img:
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=85)
                return True
        except Exception as e:
            logging.error(f"Thumbnail creation failed for {image_path}: {e}")
            return False
