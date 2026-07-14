"""File and path helper functions."""
import os
import logging
from typing import List
from common.exceptions import PreprocessingError

logger = logging.getLogger(__name__)

def ensure_dir(path: str) -> None:
    """Ensures a directory exists, creating it and parent directories if necessary.
    
    Args:
        path: Path to the directory.
        
    Raises:
        PreprocessingError: If directory cannot be created.
    """
    try:
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logger.debug(f"Created directory: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise PreprocessingError(f"Could not create directory {path}: {e}")

def find_files(directory: str, extensions: List[str]) -> List[str]:
    """Recursively searches a directory for files matching specified extensions.
    
    Args:
        directory: The search root directory.
        extensions: A list of file extensions (e.g. ['.mp4', '.avi']).
        
    Returns:
        A sorted list of absolute or resolved matching file paths.
    """
    if not os.path.exists(directory):
        logger.warning(f"Search directory does not exist: {directory}")
        return []
        
    matching_files = []
    # Normalize extensions to lowercase
    exts = [ext.lower() for ext in extensions]
    
    for root, _, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in exts:
                matching_files.append(os.path.abspath(os.path.join(root, file)))
                
    return sorted(matching_files)
