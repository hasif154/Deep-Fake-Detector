"""Video helper functions."""
import os
import logging
from typing import Dict, Any
import cv2

logger = logging.getLogger(__name__)

def get_video_info(path: str) -> Dict[str, Any]:
    """Retrieves metadata from a video file using OpenCV.
    
    Args:
        path: Path to the video file.
        
    Returns:
        Dict[str, Any]: A dictionary containing keys: 'fps', 'frame_count', 'width', 'height', 'duration_seconds', 'valid'.
    """
    info = {
        "fps": 0.0,
        "frame_count": 0,
        "width": 0,
        "height": 0,
        "duration_seconds": 0.0,
        "valid": False
    }
    
    if not os.path.exists(path):
        logger.warning(f"Video file not found: {path}")
        return info
        
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.warning(f"Could not open video file: {path}")
            return info
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        info["fps"] = fps
        info["frame_count"] = frame_count
        info["width"] = width
        info["height"] = height
        if fps > 0:
            info["duration_seconds"] = frame_count / fps
        info["valid"] = frame_count > 0 and width > 0 and height > 0
    except Exception as e:
        logger.error(f"Error querying video info for {path}: {e}")
    finally:
        if cap is not None:
            cap.release()
            
    return info
