"""Script to run the preprocessing pipeline."""
import os
import sys
import logging

# Append project root directory to path to support imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common import setup_logging, load_config
from preprocessing.pipeline import run_preprocessing_pipeline

logger = logging.getLogger("scripts.prepare_dataset")

def run_prepare(config_dir: str = "config") -> None:
    """Executes the dataset preparation pipeline.
    
    Args:
        config_dir: Directory containing YAML configuration files.
    """
    try:
        # Resolve config directory path absolute
        abs_config_dir = os.path.abspath(config_dir)
        
        # Initialize logging first using config path
        setup_logging(os.path.join(abs_config_dir, "logging.yaml"))
        
        logger.info("Initializing dataset preparation workflow...")
        
        # Load all config parameters
        config = load_config(abs_config_dir)
        
        # Execute processing
        run_preprocessing_pipeline(config)
        
    except Exception as e:
        logger.critical(f"Dataset preparation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract faces from raw videos.")
    parser.add_argument(
        "--config-dir", 
        type=str, 
        default="config", 
        help="Path to configuration directory containing split yaml files."
    )
    args = parser.parse_args()
    
    try:
        run_prepare(config_dir=args.config_dir)
    except Exception:
        sys.exit(1)
