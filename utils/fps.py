"""
FPS Counter - Measures how many frames per second
Like a SPEEDOMETER for video!
"""

import time


class FPSCounter:
    """
    Calculates frames per second in real-time.
    Simple and fast!
    """
    
    def __init__(self, window_size=30):
        """
        Initialize FPS counter
        
        Input:
            window_size = How many frames to average (larger = smoother but slower)
        """
        self.window_size = window_size
        self.timestamps = []
        self.start_time = time.time()
    
    def update(self):
        """Call this once per frame"""
        current_time = time.time()
        self.timestamps.append(current_time)
        
        # Keep only last N timestamps
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
    
    def get_fps(self):
        """
        Get current FPS
        
        Output:
            Current frames per second (float)
        """
        if len(self.timestamps) < 2:
            return 0.0
        
        # Calculate time difference
        time_diff = self.timestamps[-1] - self.timestamps[0]
        
        if time_diff <= 0:
            return 0.0
        
        # FPS = number of frames / time
        fps = (len(self.timestamps) - 1) / time_diff
        
        return fps
    
    def get_average_fps(self):
        """Get average FPS since start"""
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0
        
        # Will be calculated from actual frame processing
        return 0.0