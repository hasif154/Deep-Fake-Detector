"""Module for face detection using MediaPipe."""
import logging
from typing import Tuple, Optional
import numpy as np
import mediapipe as mp
from common.exceptions import FaceDetectionError

logger = logging.getLogger(__name__)

class FaceDetector:
    """Wraps MediaPipe Face Detection and manages the underlying resources."""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """Initializes the MediaPipe Face Detection tool.
        
        Args:
            min_detection_confidence: Minimum confidence score for detection.
            
        Raises:
            FaceDetectionError: If MediaPipe fails to initialize.
        """
        self.min_detection_confidence = min_detection_confidence
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short range (faces within 2 meters)
                min_detection_confidence=self.min_detection_confidence
            )
            logger.info("MediaPipe Face Detection initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize MediaPipe Face Detection: {e}")
            raise FaceDetectionError(f"MediaPipe initialization error: {e}")

    def detect_largest_face(self, image: np.ndarray, min_face_size: int = 30) -> Optional[Tuple[int, int, int, int]]:
        """Detects all faces in an RGB image and returns the bounding box of the largest face.
        
        Args:
            image: RGB image represented as a NumPy array.
            min_face_size: Minimum pixel dimension (width and height) for a face to be considered.
            
        Returns:
            Optional[Tuple[int, int, int, int]]: Bounding box (xmin, ymin, width, height) in pixel coordinates,
            or None if no face is detected or matches thresholds.
        """
        if image is None or image.size == 0:
            logger.error("Invalid image provided to detect_largest_face.")
            return None

        h, w, _ = image.shape
        try:
            # Process the RGB image
            results = self.face_detection.process(image)
            
            if not results.detections:
                return None
                
            largest_box = None
            largest_area = 0
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert normalized coordinates to absolute pixels
                xmin = int(bbox.xmin * w)
                ymin = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Clip values to image bounds
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                width = min(width, w - xmin)
                height = min(height, h - ymin)
                
                # Skip if width or height falls below threshold
                if width < min_face_size or height < min_face_size:
                    logger.debug(f"Skipped detected face below min_face_size: {width}x{height} < {min_face_size}")
                    continue
                    
                area = width * height
                if area > largest_area:
                    largest_area = area
                    largest_box = (xmin, ymin, width, height)
                    
            return largest_box
            
        except Exception as e:
            logger.error(f"Error during MediaPipe face detection processing: {e}")
            return None

    def close(self) -> None:
        """Closes resources allocated by MediaPipe Face Detection."""
        if hasattr(self, 'face_detection') and self.face_detection is not None:
            try:
                self.face_detection.close()
                logger.info("MediaPipe Face Detection resources released.")
            except Exception as e:
                logger.error(f"Error closing MediaPipe Face Detection: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
