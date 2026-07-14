"""Abstract base class for deepfake detection models."""
from abc import ABC, abstractmethod
from typing import Any, Dict
import tensorflow as tf

class BaseModel(ABC):
    """Abstract base class defining interface for model builders."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initializes the model builder.
        
        Args:
            config: Complete configuration dictionary.
        """
        self.config = config

    @abstractmethod
    def build(self) -> tf.keras.Model:
        """Builds and compiles the Keras model.
        
        Returns:
            tf.keras.Model: Compiled Keras model instance.
        """
        pass
