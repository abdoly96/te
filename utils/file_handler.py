import os
import time
import zipfile
import shutil
import logging
from pathlib import Path
import mimetypes
from werkzeug.utils import secure_filename

class FileHandler:
    """Handle file operations for the YOLOv8 training application"""
    
    def __init__(self):
        self.supported_image_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
        self.supported_archive_formats = ['.zip', '.rar', '.7z']
        self.max_file_size = 50 * 1024 * 1024  # 50MB per file
        self.max_archive_size = 500 * 1024 * 1024  # 500MB for archives
        
    def extract_zip(self, zip_path, extract_to_dir):
        """Extract ZIP file and return count of extracted files"""
        try:
            extracted_count = 0
            
            # Validate ZIP file
            if not zipfile.is_zipfile(zip_path):
                logging.error(f"Invalid ZIP file: {zip_path}")
                return 0
            
            # Check ZIP file size
            zip_size = os.path.getsize(zip_path)
            if zip_size > self.max_archive_size:
                logging.error(f"ZIP file too large: {zip_size} bytes")
                return 0
            
            # Create extraction directory if it doesn't exist
            os.makedirs(extract_to_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files in ZIP
                file_list = zip_ref.namelist()
                
                # Filter out directories and hidden files
                files_to_extract = [f for f in file_list 
                                  if not f.endswith('/') and not f.startswith('.') 
                                  and not f.startswith('__MACOSX/')]
                
                for file_info in zip_ref.infolist():
                    if file_info.filename in files_to_extract:
                        # Security check: prevent directory traversal
                        if self._is_safe_path(file_info.filename):
                            # Extract file
                            try:
                                zip_ref.extract(file_info, extract_to_dir)
                                
                                # Get the extracted file path
                                extracted_file_path = os.path.join(extract_to_dir, file_info.filename)
                                
                                # Move file to root of extraction directory if it's in a subdirectory
                                if os.path.dirname(file_info.filename):
                                    filename = os.path.basename(file_info.filename)
                                    new_path = os.path.join(extract_to_dir, filename)
                                    
                                    # Ensure unique filename
                                    counter = 1
                                    base_name, ext = os.path.splitext(filename)
                                    while os.path.exists(new_path):
                                        new_filename = f"{base_name}_{counter}{ext}"
                                        new_path = os.path.join(extract_to_dir, new_filename)
                                        counter += 1
                                    
                                    shutil.move(extracted_file_path, new_path)
                                    
                                    # Remove empty directories
                                    try:
                                        dir_path = os.path.dirname(extracted_file_path)
                                        if dir_path and os.path.exists(dir_path):
                                            os.rmdir(dir_path)
                                    except OSError:
                                        pass  # Directory not empty or doesn't exist
                                
                                extracted_count += 1
                                logging.info(f"Extracted: {file_info.filename}")
                                
                            except Exception as e:
                                logging.warning(f"Failed to extract {file_info.filename}: {e}")
                        else:
                            logging.warning(f"Unsafe path detected, skipping: {file_info.filename}")
            
            logging.info(f"Extraction complete: {extracted_count} files extracted from {zip_path}")
            return extracted_count
            
        except zipfile.BadZipFile:
            logging.error(f"Corrupted ZIP file: {zip_path}")
            return 0
        except Exception as e:
            logging.error(f"ZIP extraction failed: {e}")
            return 0
    
    def _is_safe_path(self, path):
        """Check if the path is safe (no directory traversal)"""
        # Normalize the path and check for directory traversal attempts
        normalized = os.path.normpath(path)
        return not (normalized.startswith('/') or 
                   normalized.startswith('\\') or 
                   '..' in normalized or
                   ':' in normalized)
    
    def validate_file_type(self, filename):
        """Validate if file type is supported"""
        _, ext = os.path.splitext(filename.lower())
        return ext in self.supported_image_formats
    
    def validate_file_size(self, file_path):
        """Validate file size"""
        try:
            file_size = os.path.getsize(file_path)
            return file_size <= self.max_file_size
        except OSError:
            return False
    
    def get_file_type(self, filename):
        """Get file type category"""
        _, ext = os.path.splitext(filename.lower())
        
        if ext in self.supported_image_formats:
            return 'image'
        elif ext in self.supported_archive_formats:
            return 'archive'
        else:
            return 'other'
    
    def get_mime_type(self, filename):
        """Get MIME type of file"""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'
    
    def sanitize_filename(self, filename):
        """Sanitize filename for safe storage"""
        # Use werkzeug's secure_filename
        secure_name = secure_filename(filename)
        
        # Additional sanitization
        secure_name = secure_name.replace(' ', '_')
        secure_name = secure_name[:100]  # Limit length
        
        return secure_name
    
    def create_unique_filename(self, directory, filename):
        """Create a unique filename in the given directory"""
        base_name, ext = os.path.splitext(filename)
        counter = 1
        unique_filename = filename
        
        while os.path.exists(os.path.join(directory, unique_filename)):
            unique_filename = f"{base_name}_{counter}{ext}"
            counter += 1
        
        return unique_filename
    
    def move_file(self, source_path, destination_path):
        """Move file from source to destination"""
        try:
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Move file
            shutil.move(source_path, destination_path)
            logging.info(f"File moved: {source_path} -> {destination_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to move file {source_path} to {destination_path}: {e}")
            return False
    
    def copy_file(self, source_path, destination_path):
        """Copy file from source to destination"""
        try:
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(source_path, destination_path)
            logging.info(f"File copied: {source_path} -> {destination_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to copy file {source_path} to {destination_path}: {e}")
            return False
    
    def delete_file(self, file_path):
        """Safely delete a file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"File deleted: {file_path}")
                return True
            else:
                logging.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logging.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    def delete_directory(self, dir_path):
        """Safely delete a directory and its contents"""
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logging.info(f"Directory deleted: {dir_path}")
                return True
            else:
                logging.warning(f"Directory not found for deletion: {dir_path}")
                return False
                
        except Exception as e:
            logging.error(f"Failed to delete directory {dir_path}: {e}")
            return False
    
    def get_directory_size(self, dir_path):
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(file_path)
                    except OSError:
                        pass  # Skip files that can't be accessed
        except Exception as e:
            logging.error(f"Error calculating directory size for {dir_path}: {e}")
        
        return total_size
    
    def get_file_count(self, dir_path, file_types=None):
        """Get count of files in directory, optionally filtered by type"""
        file_count = 0
        try:
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    if file_types is None:
                        file_count += 1
                    else:
                        file_type = self.get_file_type(filename)
                        if file_type in file_types:
                            file_count += 1
        except Exception as e:
            logging.error(f"Error counting files in {dir_path}: {e}")
        
        return file_count
    
    def cleanup_old_files(self, directory, max_age_days=30):
        """Clean up old files in directory"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            deleted_count = 0
            
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        if self.delete_file(file_path):
                            deleted_count += 1
            
            logging.info(f"Cleanup complete: {deleted_count} old files deleted from {directory}")
            return deleted_count
            
        except Exception as e:
            logging.error(f"Error during cleanup of {directory}: {e}")
            return 0
    
    def organize_files_by_type(self, source_dir, target_dir):
        """Organize files in subdirectories by type"""
        try:
            organized_count = 0
            
            for filename in os.listdir(source_dir):
                source_path = os.path.join(source_dir, filename)
                
                if os.path.isfile(source_path):
                    file_type = self.get_file_type(filename)
                    
                    # Create subdirectory for file type
                    type_dir = os.path.join(target_dir, file_type + 's')
                    os.makedirs(type_dir, exist_ok=True)
                    
                    # Move file to type-specific directory
                    target_path = os.path.join(type_dir, filename)
                    target_path = os.path.join(type_dir, self.create_unique_filename(type_dir, filename))
                    
                    if self.move_file(source_path, target_path):
                        organized_count += 1
            
            logging.info(f"Organization complete: {organized_count} files organized")
            return organized_count
            
        except Exception as e:
            logging.error(f"Error organizing files: {e}")
            return 0
