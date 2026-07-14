"""Tests for the configuration system."""
import os
import sys

# Ensure project root is in pythonpath
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
from common import load_config
from common.exceptions import ConfigurationError

def test_load_config():
    """Verifies that split configs are loaded and merged successfully."""
    config_dir = os.path.join(project_root, "config")
    config = load_config(config_dir)
    
    assert isinstance(config, dict)
    assert "experiment_name" in config
    assert "paths" in config
    assert "preprocessing" in config
    assert "training" in config
    assert "inference" in config
    
    # Check that paths under 'paths' are resolved to absolute paths
    paths = config["paths"]
    assert os.path.isabs(paths["dataset_root"])
    assert os.path.isabs(paths["models_root"])
    assert os.path.isabs(paths["outputs_root"])
