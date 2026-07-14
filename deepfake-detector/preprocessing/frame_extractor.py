"""Module for extracting frames from video files."""
import logging
from typing import List, Tuple
import cv2
import numpy as np
from common.exceptions import FrameExtractionError

logger = logging.getLogger(__name__)

def extract_frames(video_path: str, frame_skip: int = 10) -> List[Tuple[int, np.ndarray]]:
    """Extracts frames from a video at a specified step interval (frame_skip).
    
    Args:
        video_path: Path to the video file.
        frame_skip: Interval step to extract frames (extract every Nth frame).
        
    Returns:
        List[Tuple[int, np.ndarray]]: A list of tuples, where each tuple contains
        the index of the extracted frame and the frame image in RGB format.
        
    Raises:
        FrameExtractionError: If the video cannot be opened or is completely invalid.
    """
    if frame_skip <= 0:
        raise ValueError(f"frame_skip must be a positive integer, got {frame_skip}")
        
    logger.info(f"Starting frame extraction for video: {video_path} (step: {frame_skip})")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FrameExtractionError(f"Could not open video file: {video_path}")
        
    extracted_frames: List[Tuple[int, np.ndarray]] = []
    frame_idx = 0
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            if frame_idx % frame_skip == 0:
                if frame is None or frame.size == 0:
                    logger.warning(f"Frame {frame_idx} in video {video_path} is empty or corrupted, skipping.")
                else:
                    try:
                        # Convert BGR to RGB format
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        extracted_frames.append((frame_idx, rgb_frame))
                    except Exception as parse_err:
                        logger.warning(f"Failed to process frame {frame_idx} in {video_path}: {parse_err}")
                
            frame_idx += 1
            
        logger.info(f"Successfully extracted {len(extracted_frames)} frames from {video_path}")
    except Exception as e:
        logger.error(f"Unexpected error during frame extraction for {video_path}: {e}")
    finally:
        cap.release()
        
    return extracted_frames
