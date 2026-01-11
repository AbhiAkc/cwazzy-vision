"""
Face Tracker - Gives IDs to faces so we can track them
Like giving each person a NAME so we remember who they are!
"""

import numpy as np
from config import MAX_TRACKING_DISTANCE, TRACKING_MEMORY_FRAMES


class FaceTracker:
    """
    Simple centroid-based face tracker.
    Assigns IDs to faces and tracks them across frames.
    """
    
    def __init__(self):
        """Initialize tracker"""
        self.tracked_faces = {}  # Dictionary to store tracked faces
        self.next_id = 0
        self.frame_count = 0
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Input:
            detections = List of (x, y, w, h, confidence)
        """
        self.frame_count += 1
        
        # Calculate centroids of new detections
        new_centroids = []
        for (x, y, w, h, conf) in detections:
            cx = x + w // 2
            cy = y + h // 2
            new_centroids.append((cx, cy, x, y, w, h, conf))
        
        # If no tracked faces yet, assign new IDs
        if not self.tracked_faces:
            for idx, (cx, cy, x, y, w, h, conf) in enumerate(new_centroids):
                self.tracked_faces[self.next_id] = {
                    'centroid': (cx, cy),
                    'box': (x, y, w, h),
                    'confidence': conf,
                    'last_seen': self.frame_count,
                    'age': 1
                }
                self.next_id += 1
            return
        
        # Match new detections to existing tracks
        used_ids = set()
        used_detections = set()
        
        for face_id, face_data in list(self.tracked_faces.items()):
            old_cx, old_cy = face_data['centroid']
            min_distance = float('inf')
            best_match = -1
            
            # Find closest new detection
            for idx, (cx, cy, x, y, w, h, conf) in enumerate(new_centroids):
                if idx in used_detections:
                    continue
                
                distance = np.sqrt((cx - old_cx) ** 2 + (cy - old_cy) ** 2)
                
                if distance < min_distance and distance < MAX_TRACKING_DISTANCE:
                    min_distance = distance
                    best_match = idx
            
            # Update or remove face
            if best_match >= 0:
                cx, cy, x, y, w, h, conf = new_centroids[best_match]
                self.tracked_faces[face_id]['centroid'] = (cx, cy)
                self.tracked_faces[face_id]['box'] = (x, y, w, h)
                self.tracked_faces[face_id]['confidence'] = conf
                self.tracked_faces[face_id]['last_seen'] = self.frame_count
                self.tracked_faces[face_id]['age'] += 1
                used_ids.add(face_id)
                used_detections.add(best_match)
            else:
                # Face not found this frame
                frames_missing = self.frame_count - face_data['last_seen']
                
                if frames_missing > TRACKING_MEMORY_FRAMES:
                    # Face is gone - remove it
                    del self.tracked_faces[face_id]
        
        # Add new detections that weren't matched
        for idx, (cx, cy, x, y, w, h, conf) in enumerate(new_centroids):
            if idx not in used_detections:
                self.tracked_faces[self.next_id] = {
                    'centroid': (cx, cy),
                    'box': (x, y, w, h),
                    'confidence': conf,
                    'last_seen': self.frame_count,
                    'age': 1
                }
                self.next_id += 1
    
    def get_id(self, detection_index):
        """
        Get ID for a detection (simple approach)
        In real implementation, would use proper matching
        """
        # For demo, just use index
        face_ids = list(self.tracked_faces.keys())
        if detection_index < len(face_ids):
            return face_ids[detection_index]
        return -1
    
    def get_count(self):
        """Get number of currently tracked faces"""
        return len(self.tracked_faces)