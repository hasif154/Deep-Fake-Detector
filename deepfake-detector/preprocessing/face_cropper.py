"""Module for cropping face regions from images."""
import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)

def crop_face(image: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = 0.15) -> np.ndarray:
    """Crops a face from an image using a bounding box, applying a padding percentage.
    
    Args:
        image: Original RGB image represented as a NumPy array.
        bbox: Bounding box tuple (xmin, ymin, width, height) in pixel coordinates.
        padding: Padding fraction to add around the face (e.g. 0.15 adds 15% padding).
        
    Returns:
        np.ndarray: The cropped face image.
        
    Raises:
        ValueError: If the input image is empty or invalid.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot crop face from empty or invalid image.")
        
    h, w, _ = image.shape
    xmin, ymin, width, height = bbox
    
    # Calculate padding values
    pad_w = int(width * padding)
    pad_h = int(height * padding)
    
    # Define cropped region coordinates with padding
    crop_xmin = max(0, xmin - pad_w)
    crop_ymin = max(0, ymin - pad_h)
    crop_xmax = min(w, xmin + width + pad_w)
    crop_ymax = min(h, ymin + height + pad_h)
    
    logger.debug(f"Applied crop coordinates: x[{crop_xmin}:{crop_xmax}], y[{crop_ymin}:{crop_ymax}]")
    
    return image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
