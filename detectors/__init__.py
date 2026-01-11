
"""
Detectors package - Face detection methods
"""

from .haar_detector import HaarDetector
from .dnn_detector import DNNDetector

__all__ = ['HaarDetector', 'DNNDetector']