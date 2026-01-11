"""
Haar Cascade Face Detector
SIMPLE METHOD: Like showing pictures of faces to a robot.
The robot learns what faces look like and finds them.
Fast but sometimes makes mistakes (false positives).
"""

import cv2
import numpy as np
from config import (
    HAAR_SCALE_FACTOR,
    HAAR_MIN_NEIGHBORS,
    HAAR_MIN_SIZE
)


class HaarDetector:
    """
    Haar Cascade detector class.
    Think of it like a PATTERN MATCHER.
    """
    
    def __init__(self, cascade_path):
        """
        Initialize Haar Cascade detector
        Input: Path to cascade XML file
        """
        print(f"Loading Haar Cascade from: {cascade_path}")
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.cascade.empty():
            raise Exception(f"❌ Could not load Haar Cascade from {cascade_path}")
        
        print("✅ Haar Cascade loaded successfully!")
    
    def detect(self, frame):
        """
        Detect faces in frame using Haar Cascade
        
        Input: 
            frame = Image/video frame
        
        Output:
            List of (x, y, width, height, confidence)
            where confidence is always 1.0 for Haar
        """
        
        # Convert to grayscale (Haar works better on grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram (makes contrast better)
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to our standard format: (x, y, w, h, confidence)
        # Haar doesn't give confidence, so we use 1.0
        detections = []
        for (x, y, w, h) in faces:
            detections.append((x, y, w, h, 1.0))
        
        return detections