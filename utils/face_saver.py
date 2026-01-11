"""
Face Saver - Saves detected faces to disk
Like taking SCREENSHOTS of faces!
"""

import cv2
import os
from datetime import datetime
from pathlib import Path
from config import CAPTURED_FACES_DIR


class FaceSaver:
    """
    Saves detected face crops to disk with timestamps.
    """
    
    def __init__(self):
        """Initialize face saver"""
        # Create output directory if it doesn't exist
        Path(CAPTURED_FACES_DIR).mkdir(parents=True, exist_ok=True)
        self.save_count = 0
    
    def save(self, face_roi, confidence=0.0):
        """
        Save a face crop to disk
        
        Input:
            face_roi = Face region (cropped image)
            confidence = Confidence score (optional)
        
        Output:
            Path to saved file
        """
        # Create timestamp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Format confidence in filename
        conf_str = f"_conf{confidence:.2f}" if confidence > 0 else ""
        
        filename = f"face_{timestamp}{conf_str}.jpg"
        filepath = os.path.join(CAPTURED_FACES_DIR, filename)
        
        # Save the face
        try:
            cv2.imwrite(filepath, face_roi)
            self.save_count += 1
            print(f"✅ Saved: {filename} ({face_roi.shape[0]}x{face_roi.shape[1]} pixels)")
            return filepath
        except Exception as e:
            print(f"❌ Error saving face: {e}")
            return None
    
    def get_save_count(self):
        """Get total number of faces saved"""
        return self.save_count