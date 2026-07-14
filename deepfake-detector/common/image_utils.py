"""Image helper functions."""
import os
import logging
import cv2
import numpy as np
from common.file_utils import ensure_dir

logger = logging.getLogger(__name__)

def load_image(path: str) -> np.ndarray:
    """Loads an image using OpenCV and converts it from BGR to RGB format.
    
    Args:
        path: Path to the image file.
        
    Returns:
        np.ndarray: The loaded image in RGB format.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as an image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
        
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image: {path} (possibly corrupted or unsupported format)")
        
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_image(image: np.ndarray, path: str) -> bool:
    """Saves an image in RGB format to a file path, converting it to BGR first.
    
    Args:
        image: np.ndarray representing the RGB image.
        path: Target file path.
        
    Returns:
        bool: True if the image was successfully saved, False otherwise.
    """
    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            ensure_dir(dir_name)
        # Convert RGB to BGR for OpenCV write
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(path, bgr_image)
        if not success:
            logger.error(f"OpenCV failed to write image to {path}")
        return success
    except Exception as e:
        logger.error(f"Error saving image to {path}: {e}")
        return False
