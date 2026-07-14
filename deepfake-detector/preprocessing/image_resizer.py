"""Module for resizing images to target dimensions."""
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def resize_image(image: np.ndarray, target_width: int = 224, target_height: int = 224) -> np.ndarray:
    """Resizes an image to the target width and height using cv2.INTER_AREA.
    
    Args:
        image: RGB image represented as a NumPy array.
        target_width: Target width in pixels.
        target_height: Target height in pixels.
        
    Returns:
        np.ndarray: The resized image.
        
    Raises:
        ValueError: If input image or dimensions are invalid.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot resize empty or invalid image.")
        
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Target width and height must be positive, got {target_width}x{target_height}")
        
    h, w, _ = image.shape
    if h == target_height and w == target_width:
        return image
        
    logger.debug(f"Resizing image from {w}x{h} to {target_width}x{target_height}")
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
