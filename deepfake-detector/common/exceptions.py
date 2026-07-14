"""Custom exceptions for the Hybrid Deepfake Detection project."""

class DeepfakeDetectorException(Exception):
    """Base exception for all errors in the deepfake detection system."""
    pass

class ConfigurationError(DeepfakeDetectorException):
    """Raised when there is an error in configuration loading or validation."""
    pass

class PreprocessingError(DeepfakeDetectorException):
    """Raised when a preprocessing component fails."""
    pass

class FrameExtractionError(PreprocessingError):
    """Raised when frame extraction fails for a video."""
    pass

class FaceDetectionError(PreprocessingError):
    """Raised when face detection errors occur."""
    pass

class ValidationError(PreprocessingError):
    """Raised when image validation fails."""
    pass
