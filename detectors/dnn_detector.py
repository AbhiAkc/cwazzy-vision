"""
DNN (Deep Neural Network) Face Detector
SMART METHOD: Like a HUMAN looking at faces.
It's more accurate but slower than Haar.
Uses SSD (Single Shot Detector) with ResNet backbone.
"""

import cv2
import numpy as np
from config import DNN_CONFIDENCE_THRESHOLD


class DNNDetector:
    """
    Deep Neural Network face detector.
    Uses pre-trained SSD model for high accuracy.
    """
    
    def __init__(self, proto_path, model_path):
        """
        Initialize DNN detector
        
        Input:
            proto_path = Path to deploy.prototxt
            model_path = Path to res10_300x300_ssd.caffemodel
        """
        print(f"Loading DNN from: {proto_path}, {model_path}")
        
        # Load pre-trained model
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        
        print("✅ DNN model loaded successfully!")
    
    def detect(self, frame):
        """
        Detect faces using DNN
        
        Input:
            frame = Image/video frame
        
        Output:
            List of (x, y, width, height, confidence)
        """
        
        h, w = frame.shape[:2]
        
        # Create a blob (network-ready image)
        # The model expects 300x300 images
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=(300, 300),
            mean=[104.0, 177.0, 123.0],  # BGR means (these are magic numbers!)
            swapRB=False,
            crop=False
        )
        
        # Run detection
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Process detections
        faces = []
        
        # detections shape: (1, 1, N, 7)
        # Each detection: [image_id, label, confidence, x1, y1, x2, y2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter by confidence threshold
            if confidence < DNN_CONFIDENCE_THRESHOLD:
                continue
            
            # Get bounding box coordinates
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            
            # Convert to (x, y, width, height) format
            x = x1
            y = y1
            width = x2 - x1
            height = y2 - y1
            
            # Skip invalid detections
            if width <= 0 or height <= 0:
                continue
            
            faces.append((x, y, width, height, confidence))
        
        return faces