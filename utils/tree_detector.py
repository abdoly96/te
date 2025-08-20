import cv2
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime

class TreeDetector:
    """Automatic tree detection using computer vision techniques"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_area = 1000  # Minimum area for tree detection
        self.max_area = 50000  # Maximum area for tree detection

    def detect_trees(self, image_path):
        """
        Detect trees in an image using color and shape analysis
        Returns list of bounding boxes in YOLO format
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return []

            # Convert to different color spaces for better tree detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            # Create masks for different tree characteristics
            tree_masks = self._create_tree_masks(hsv, lab)

            # Combine masks
            combined_mask = self._combine_masks(tree_masks)

            # Find contours
            contours = self._find_tree_contours(combined_mask)

            # Filter and convert to bounding boxes
            bounding_boxes = self._filter_and_convert_contours(contours, image.shape)

            self.logger.info(f"Detected {len(bounding_boxes)} trees in {image_path}")
            return bounding_boxes

        except Exception as e:
            self.logger.error(f"Error detecting trees in {image_path}: {e}")
            return []

    def _create_tree_masks(self, hsv, lab):
        """Create multiple masks for different tree characteristics"""
        masks = []

        # Green color range for trees (HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        masks.append(green_mask)

        # Dark green/brown for tree trunks (LAB)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([255, 150, 150])
        dark_mask = cv2.inRange(lab, lower_dark, upper_dark)
        masks.append(dark_mask)

        # Brightness mask for leaves
        gray = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        masks.append(bright_mask)

        return masks

    def _combine_masks(self, masks):
        """Combine multiple masks using logical operations"""
        if not masks:
            return np.zeros((100, 100), dtype=np.uint8)

        combined = masks[0]
        for mask in masks[1:]:
            combined = cv2.bitwise_or(combined, mask)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        return combined

    def _find_tree_contours(self, mask):
        """Find contours in the mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _filter_and_convert_contours(self, contours, image_shape):
        """Filter contours and convert to YOLO format bounding boxes"""
        height, width = image_shape[:2]
        bounding_boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio (trees are usually taller than wide)
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                continue

            # Convert to YOLO format (normalized coordinates)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_width = w / width
            norm_height = h / height

            # Ensure coordinates are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            norm_width = max(0, min(1, norm_width))
            norm_height = max(0, min(1, norm_height))

            bounding_boxes.append({
                'class_id': 0,  # Class ID for palm tree
                'x_center': x_center,
                'y_center': y_center,
                'width': norm_width,
                'height': norm_height,
                'confidence': 0.8  # Default confidence
            })

        return bounding_boxes

    def process_image_batch(self, image_folder, output_folder):
        """
        Process a batch of images and save annotated versions
        Returns statistics about processing
        """
        if not os.path.exists(image_folder):
            self.logger.error(f"Image folder not found: {image_folder}")
            return {'error': 'Image folder not found'}

        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
            image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))

        if not image_files:
            return {'error': 'No image files found'}

        processed_count = 0
        total_trees = 0
        failed_images = []

        for image_path in image_files:
            try:
                # Detect trees
                bounding_boxes = self.detect_trees(str(image_path))

                if bounding_boxes:
                    # Save annotated image
                    annotated_image = self._draw_annotations(str(image_path), bounding_boxes)
                    output_path = os.path.join(output_folder, f"annotated_{image_path.name}")
                    cv2.imwrite(output_path, annotated_image)

                    # Save YOLO format annotations
                    annotation_path = os.path.join(output_folder, f"{image_path.stem}.txt")
                    self._save_yolo_annotations(annotation_path, bounding_boxes)

                    processed_count += 1
                    total_trees += len(bounding_boxes)

                    self.logger.info(f"Processed {image_path.name}: {len(bounding_boxes)} trees")
                else:
                    # Save original image if no trees detected
                    output_path = os.path.join(output_folder, image_path.name)
                    cv2.imwrite(output_path, cv2.imread(str(image_path)))

            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                failed_images.append(str(image_path))

        # Create classes.txt file
        classes_path = os.path.join(output_folder, 'classes.txt')
        with open(classes_path, 'w') as f:
            f.write('palm tree\n')

        return {
            'processed_images': processed_count,
            'total_trees_detected': total_trees,
'failed_images': failed_images,
            'output_folder': output_folder
        }

    def _draw_annotations(self, image_path, bounding_boxes):
        """Draw bounding boxes on the image"""
        image = cv2.imread(image_path)
        if image is None:
            return None

        height, width = image.shape[:2]

        for box in bounding_boxes:
            # Convert normalized coordinates back to pixel coordinates
            x_center = int(box['x_center'] * width)
            y_center = int(box['y_center'] * height)
            w = int(box['width'] * width)
            h = int(box['height'] * height)

            # Calculate top-left and bottom-right coordinates
            x1 = max(0, x_center - w // 2)
            y1 = max(0, y_center - h // 2)
            x2 = min(width, x_center + w // 2)
            y2 = min(height, y_center + h // 2)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"Palm Tree: {box['confidence']:.2f}"
            cv2.putText(image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return image

    def _save_yolo_annotations(self, annotation_path, bounding_boxes):
        """Save annotations in YOLO format"""
        with open(annotation_path, 'w') as f:
            for box in bounding_boxes:
                f.write(f"{box['class_id']} {box['x_center']:.6f} {box['y_center']:.6f} "
                       f"{box['width']:.6f} {box['height']:.6f}\n")
