"""Centralized logging configuration."""
import os
import logging
import logging.config
from typing import Optional
import yaml

def setup_logging(config_path: str = "config/logging.yaml") -> None:
    """Sets up Python standard logging with the specified config yaml.
    
    Args:
        config_path: Path to the logging configuration yaml.
    """
    if not os.path.exists(config_path):
        # Fallback to basic logging if config not found
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logger = logging.getLogger(__name__)
        logger.warning(f"Logging configuration not found at {config_path}. Falling back to basic logging.")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            log_config = yaml.safe_load(f)
            
        # Ensure log directories exist for any file handlers
        handlers = log_config.get("handlers", {})
        for handler_cfg in handlers.values():
            if isinstance(handler_cfg, dict) and "filename" in handler_cfg:
                filename = handler_cfg["filename"]
                log_dir = os.path.dirname(filename)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                    
        logging.config.dictConfig(log_config)
    except Exception as e:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load logging config from {config_path}: {e}. Falling back to basic logging.")
