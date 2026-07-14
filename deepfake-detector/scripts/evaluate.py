"""Script for executing the model evaluation process."""
import os
import sys
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common import setup_logging, load_config
from training.evaluate import evaluate_model

logger = logging.getLogger(__name__)

def run_evaluation(model_path: str, test_dir: str, config_dir: str = "config") -> None:
    """Invokes the model evaluation process.
    
    Args:
        model_path: Path to model file.
        test_dir: Path to test dataset.
        config_dir: Path to config folder.
    """
    setup_logging(os.path.join(config_dir, "logging.yaml"))
    config = load_config(config_dir)
    evaluate_model(model_path, test_dir, config)

if __name__ == "__main__":
    # Skeletons for arguments
    pass
