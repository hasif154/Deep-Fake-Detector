"""Tests for dataset loading pipelines."""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
from training.dataset_loader import load_datasets

def test_load_datasets():
    """Verifies loader returns train/val/test data pipelines."""
    # Using mock parameters for config loading
    config = {
        "paths": {"dataset_root": "datasets"},
        "training": {"img_height": 224, "img_width": 224, "batch_size": 32}
    }
    train, val, test = load_datasets(config)
    assert train is not None
    assert val is not None
    assert test is not None
