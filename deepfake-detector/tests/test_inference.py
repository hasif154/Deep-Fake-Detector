"""Tests for the inference subsystem."""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
from inference.predict_image import predict_single_image

def test_predict_single_image():
    """Validates signature return structures for single image inferences."""
    dummy_model = None
    config = {"inference": {"threshold": 0.5}}
    
    label, confidence = predict_single_image("dummy_path.jpg", dummy_model, config)
    assert label in ["REAL", "FAKE"]
    assert 0.0 <= confidence <= 1.0
