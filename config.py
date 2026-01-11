"""
Configuration file for CWAZZY VISION
Think of this as our APP SETTINGS - like volume control!
Change things here instead of digging through code.
"""

import os
from pathlib import Path

# ════════════════════════════════════════════════════════════════
# VIDEO SETTINGS
# ════════════════════════════════════════════════════════════════

# Frame dimensions (width, height)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Target FPS (frames per second) - how smooth the video should be
FPS_TARGET = 60

# Skip frames for faster processing
# Higher number = faster but less responsive
SKIP_FRAMES = 1  # Process every frame

# Scale factor for detection (0.5 = process at 50% size = 2x faster)
SCALE_FACTOR = 0.5

# ════════════════════════════════════════════════════════════════
# FILE PATHS
# ════════════════════════════════════════════════════════════════

# Base directory
BASE_DIR = Path(__file__).parent

# Haar Cascade model path
HAAR_CASCADE_PATH = str(BASE_DIR / "models" / "haarcascade_frontalface_default.xml")

# DNN model files
DNN_PROTO_PATH = str(BASE_DIR / "models" / "deploy.prototxt")
DNN_MODEL_PATH = str(BASE_DIR / "models" / "res10_300x300_ssd.caffemodel")

# Output directory for captured faces
CAPTURED_FACES_DIR = str(BASE_DIR / "data" / "captured_faces")

# ════════════════════════════════════════════════════════════════
# DETECTION PARAMETERS
# ════════════════════════════════════════════════════════════════

# Haar Cascade settings
HAAR_SCALE_FACTOR = 1.1  # How much image size is reduced at each step
HAAR_MIN_NEIGHBORS = 5   # How many neighbors each rectangle should have
HAAR_MIN_SIZE = (30, 30) # Minimum face size

# DNN settings
DNN_CONFIDENCE_THRESHOLD = 0.5  # Only faces with >= 50% confidence

# ════════════════════════════════════════════════════════════════
# FACE TRACKING
# ════════════════════════════════════════════════════════════════

# Maximum distance for face matching (in pixels)
MAX_TRACKING_DISTANCE = 100

# How many frames to keep tracking a missing face
TRACKING_MEMORY_FRAMES = 30

# ════════════════════════════════════════════════════════════════
# UI SETTINGS
# ════════════════════════════════════════════════════════════════

# Show FPS counter
SHOW_FPS = True

# Show confidence scores
SHOW_CONFIDENCE = True

# Color for bounding boxes (BGR format - OpenCV uses BGR not RGB!)
BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (0, 255, 255)  # Yellow
ID_COLOR = (255, 0, 255)  # Magenta

# ════════════════════════════════════════════════════════════════
# DEBUG & LOGGING
# ════════════════════════════════════════════════════════════════

# Print debug information
DEBUG = False

# Log detections to file
LOG_DETECTIONS = False