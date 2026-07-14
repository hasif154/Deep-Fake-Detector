"""Pipeline orchestrator for video preprocessing."""
import os
import time
import logging
from typing import Dict, Any
from common.exceptions import PreprocessingError, FrameExtractionError
from common.file_utils import ensure_dir, find_files
from common.image_utils import save_image
from preprocessing.frame_extractor import extract_frames
from preprocessing.face_detector import FaceDetector
from preprocessing.face_cropper import crop_face
from preprocessing.validator import validate_image
from preprocessing.image_resizer import resize_image

logger = logging.getLogger(__name__)

def process_video(
    video_path: str,
    output_video_dir: str,
    detector: FaceDetector,
    config: Dict[str, Any]
) -> Dict[str, int]:
    """Processes a single video, extracting frames, detecting/cropping faces, and saving them.
    
    Args:
        video_path: Path to the raw video file.
        output_video_dir: Directory where processed face images for this video will be saved.
        detector: An active FaceDetector instance.
        config: The complete configuration dictionary.
        
    Returns:
        Dict[str, int]: Statistics for this video processing run.
    """
    stats = {
        "frames_extracted": 0,
        "faces_detected": 0,
        "faces_skipped": 0,
        "images_saved": 0
    }
    
    prep_config = config.get("preprocessing", {})
    frame_skip = prep_config.get("frame_skip", 10)
    crop_padding = prep_config.get("crop_padding", 0.15)
    min_face_size = prep_config.get("min_face_size", 30)
    target_size = prep_config.get("target_size", {"width": 224, "height": 224})
    target_w = target_size.get("width", 224)
    target_h = target_size.get("height", 224)
    
    logger.info(f"Video started: {video_path}")
    
    try:
        frames = extract_frames(video_path, frame_skip=frame_skip)
        stats["frames_extracted"] = len(frames)
    except FrameExtractionError as e:
        logger.error(f"Failed to extract frames for {video_path}: {e}")
        return stats
    except Exception as e:
        logger.error(f"Unexpected error extracting frames for {video_path}: {e}")
        return stats
        
    # Only ensure folder creation if we extracted any frames
    if len(frames) > 0:
        ensure_dir(output_video_dir)
    
    for frame_idx, frame in frames:
        try:
            # Detect face
            bbox = detector.detect_largest_face(frame, min_face_size=min_face_size)
            if bbox is None:
                logger.debug(f"No face detected in frame {frame_idx} of {video_path}")
                stats["faces_skipped"] += 1
                continue
                
            stats["faces_detected"] += 1
            
            # Crop face
            cropped = crop_face(frame, bbox, padding=crop_padding)
            
            # Validate face image
            if not validate_image(cropped, min_width=min_face_size, min_height=min_face_size):
                logger.warning(f"Face crop in frame {frame_idx} of {video_path} failed validation.")
                stats["faces_skipped"] += 1
                continue
                
            # Resize face image
            resized = resize_image(cropped, target_width=target_w, target_height=target_h)
            
            # Save image
            output_filename = f"frame_{frame_idx:04d}.jpg"
            output_path = os.path.join(output_video_dir, output_filename)
            
            if save_image(resized, output_path):
                stats["images_saved"] += 1
                logger.debug(f"Face image saved: {output_path}")
            else:
                logger.error(f"Failed to save processed face image: {output_path}")
                stats["faces_skipped"] += 1
                
        except Exception as frame_err:
            logger.error(f"Error processing frame {frame_idx} in video {video_path}: {frame_err}")
            stats["faces_skipped"] += 1
            
    logger.info(
        f"Video processed statistics for {os.path.basename(video_path)}: "
        f"Extracted: {stats['frames_extracted']}, Detected: {stats['faces_detected']}, "
        f"Skipped: {stats['faces_skipped']}, Saved: {stats['images_saved']}"
    )
    return stats

def run_preprocessing_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the entire video preprocessing pipeline based on configuration.
    
    Processes folders matching raw classes 'real' and 'fake'.
    
    Args:
        config: The merged configuration dictionary.
        
    Returns:
        Dict[str, Any]: Summary stats of the pipeline run.
        
    Raises:
        PreprocessingError: If a general error occurs during initialization.
    """
    start_time = time.time()
    
    paths = config.get("paths", {})
    dataset_root = paths.get("dataset_root", "datasets")
    
    raw_dir = os.path.join(dataset_root, "raw")
    processed_dir = os.path.join(dataset_root, "processed")
    
    classes = ["real", "fake"]
    
    prep_config = config.get("preprocessing", {})
    supported_extensions = prep_config.get("supported_extensions", [".mp4", ".avi", ".mov", ".mkv"])
    min_det_conf = prep_config.get("min_detection_confidence", 0.5)
    
    pipeline_stats = {
        "videos_processed": 0,
        "frames_extracted": 0,
        "faces_detected": 0,
        "faces_skipped": 0,
        "images_saved": 0,
        "failed_videos": 0,
        "execution_time_seconds": 0.0
    }
    
    detector = None
    try:
        detector = FaceDetector(min_detection_confidence=min_det_conf)
        
        for cls in classes:
            class_raw_dir = os.path.join(raw_dir, cls)
            class_processed_dir = os.path.join(processed_dir, cls)
            
            if not os.path.exists(class_raw_dir):
                logger.warning(f"Raw class directory not found: {class_raw_dir}. Skipping.")
                continue
                
            video_files = find_files(class_raw_dir, supported_extensions)
            logger.info(f"Found {len(video_files)} '{cls}' videos to process in {class_raw_dir}")
            
            for video_path in video_files:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_video_dir = os.path.join(class_processed_dir, video_name)
                
                try:
                    video_stats = process_video(
                        video_path=video_path,
                        output_video_dir=output_video_dir,
                        detector=detector,
                        config=config
                    )
                    
                    pipeline_stats["videos_processed"] += 1
                    pipeline_stats["frames_extracted"] += video_stats["frames_extracted"]
                    pipeline_stats["faces_detected"] += video_stats["faces_detected"]
                    pipeline_stats["faces_skipped"] += video_stats["faces_skipped"]
                    pipeline_stats["images_saved"] += video_stats["images_saved"]
                    
                    if video_stats["frames_extracted"] == 0:
                        pipeline_stats["failed_videos"] += 1
                        
                except Exception as vid_err:
                    pipeline_stats["failed_videos"] += 1
                    logger.error(f"Error processing video file {video_path}: {vid_err}")
                    
    except Exception as e:
        logger.critical(f"Critical pipeline error: {e}")
        raise PreprocessingError(f"Pipeline execution aborted: {e}")
    finally:
        if detector is not None:
            detector.close()
            
    execution_time = time.time() - start_time
    pipeline_stats["execution_time_seconds"] = execution_time
    
    # Format and log summary stats
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    
    logger.info("=" * 50)
    logger.info("PREPROCESSING PIPELINE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Videos processed: {pipeline_stats['videos_processed']}")
    logger.info(f"Frames extracted: {pipeline_stats['frames_extracted']}")
    logger.info(f"Faces detected:   {pipeline_stats['faces_detected']}")
    logger.info(f"Faces skipped:    {pipeline_stats['faces_skipped']}")
    logger.info(f"Images saved:     {pipeline_stats['images_saved']}")
    logger.info(f"Failed videos:    {pipeline_stats['failed_videos']}")
    logger.info(f"Execution time:   {minutes} minutes and {seconds} seconds")
    logger.info("=" * 50)
    
    return pipeline_stats
