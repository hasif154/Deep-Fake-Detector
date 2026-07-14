"""Script for executing the training process."""
import os
import sys
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common import setup_logging, load_config
from training.trainer import train_model

logger = logging.getLogger(__name__)

def run_training(config_dir: str = "config") -> None:
    """Invokes the training process.
    
    Args:
        config_dir: Configuration directory.
    """
    setup_logging(os.path.join(config_dir, "logging.yaml"))
    config = load_config(config_dir)
    train_model(config)

if __name__ == "__main__":
    run_training()
