"""Configuration loader module for loading and merging split YAML configurations."""
import os
from typing import Any, Dict
import yaml
from common.exceptions import ConfigurationError

def load_config(config_dir: str = "config") -> Dict[str, Any]:
    """Loads and merges configuration parameters from split YAML files in the config folder.
    
    Args:
        config_dir: Path to the config directory containing the split yaml files.
        
    Returns:
        Dict[str, Any]: Combined configuration parameters.
        
    Raises:
        ConfigurationError: If any of the required config files are missing or invalid.
    """
    merged_config: Dict[str, Any] = {}
    config_files = ["config.yaml", "dataset.yaml", "training.yaml", "inference.yaml"]
    
    # Normalize config_dir path
    if not os.path.isabs(config_dir):
        # Resolve path relative to current working directory or search upwards
        config_dir = os.path.abspath(config_dir)
        
    for filename in config_files:
        filepath = os.path.join(config_dir, filename)
        if not os.path.exists(filepath):
            raise ConfigurationError(f"Required configuration file not found: {filepath}")
            
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and isinstance(data, dict):
                    merged_config.update(data)
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration from {filepath}: {e}")
            
    # Resolve relative paths under 'paths' relative to the project root
    # Project root is assumed to be the parent of the config directory
    project_root = os.path.abspath(os.path.join(config_dir, ".."))
    
    if "paths" in merged_config and isinstance(merged_config["paths"], dict):
        paths_dict = merged_config["paths"]
        for key, value in paths_dict.items():
            if isinstance(value, str) and not os.path.isabs(value):
                # Resolve relative path using project root
                paths_dict[key] = os.path.normpath(os.path.join(project_root, value))
                
    return merged_config
