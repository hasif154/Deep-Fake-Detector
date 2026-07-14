"""Module for validating face image structures and properties."""
import logging
import numpy as np

logger = logging.getLogger(__name__)

def validate_image(image: np.ndarray, min_width: int = 30, min_height: int = 30) -> bool:
    """Validates if a face image meets requirements.
    
    Checks:
        - Image array exists and is a numpy array.
        - Image dimensions are greater than zero.
        - Dimensions meet the minimum resolution thresholds.
        - Image has exactly 3 channels (RGB).
        - Image contains actual non-zero pixel values (not entirely black).
        
    Args:
        image: NumPy array representing the face image.
        min_width: Minimum width required.
        min_height: Minimum height required.
        
    Returns:
        bool: True if the image passes all checks, False otherwise.
    """
    if image is None:
        logger.warning("Image validation failed: Image is None.")
        return False
        
    if not isinstance(image, np.ndarray):
        logger.warning("Image validation failed: Input is not a NumPy array.")
        return False
        
    if image.size == 0:
        logger.warning("Image validation failed: NumPy array is empty.")
        return False
        
    if len(image.shape) != 3 or image.shape[2] != 3:
        logger.warning(f"Image validation failed: Shape {image.shape} does not have 3 channels.")
        return False
        
    h, w, _ = image.shape
    if w < min_width or h < min_height:
        logger.warning(f"Image validation failed: Resolution {w}x{h} is below minimum {min_width}x{min_height}.")
        return False
        
    # Check if image data is corrupted (e.g. all pixels are black)
    if np.all(image == 0):
        logger.warning("Image validation failed: Image is completely black/empty.")
        return False
        
    return True
