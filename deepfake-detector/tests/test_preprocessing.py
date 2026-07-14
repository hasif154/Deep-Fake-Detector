"""Tests for preprocessing pipeline components."""
import os
import sys
import numpy as np
import pytest

# Ensure project root is in pythonpath
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.face_cropper import crop_face
from preprocessing.validator import validate_image
from preprocessing.image_resizer import resize_image

def test_crop_face():
    """Verifies that the face cropper applies padding and bounds limits correctly."""
    # Create synthetic RGB image (100x100 pixels)
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # Bounding box coordinates: xmin, ymin, width, height
    bbox = (20, 20, 40, 40)
    
    # Crop with 10% padding
    cropped = crop_face(image, bbox, padding=0.1)
    
    # Output should be valid dimensions and channels
    assert cropped.ndim == 3
    assert cropped.shape[2] == 3
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0
    
    # Testing coordinate clamping by cropping out of bounds
    bbox_out = (80, 80, 50, 50)
    cropped_out = crop_face(image, bbox_out, padding=0.0)
    assert cropped_out.shape[0] == 20  # ymin=80 to max=100
    assert cropped_out.shape[1] == 20  # xmin=80 to max=100

def test_validate_image():
    """Validates the behavior of our validation constraints."""
    # Valid synthetic RGB image
    valid_img = np.random.randint(1, 255, (50, 50, 3), dtype=np.uint8)
    assert validate_image(valid_img, min_width=30, min_height=30) is True
    
    # Fails under minimum dimension checks
    assert validate_image(valid_img, min_width=60, min_height=60) is False
    
    # Fails for blank images (all zero pixels)
    blank_img = np.zeros((50, 50, 3), dtype=np.uint8)
    assert validate_image(blank_img) is False
    
    # Fails for incorrect number of channels
    single_channel_img = np.random.randint(1, 255, (50, 50), dtype=np.uint8)
    assert validate_image(single_channel_img) is False

def test_resize_image():
    """Ensures resizing correctly outputs targeted shape dimensions."""
    image = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
    resized = resize_image(image, target_width=224, target_height=224)
    
    assert resized.shape == (224, 224, 3)
