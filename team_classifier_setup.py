import supervision as sv
import numpy as np
from tqdm import tqdm
import logging
import config # Uses updated class IDs from config.py
import cv2
import os # Added import for os.path.exists

try:
    from sports.common.team import TeamClassifier
    TEAM_CLASSIFIER_AVAILABLE = True
except ImportError:
    logging.error("Failed to import TeamClassifier from 'sports.common.team'. Please ensure it's installed and the path is correct. Team classification will be disabled.")
    TeamClassifier = None # Define as None if import fails
    TEAM_CLASSIFIER_AVAILABLE = False


def setup_team_classifier(player_detection_model, source_video_path: str):
    """
    Collects player crops from the initial part of the video and fits the TeamClassifier.
    Processes a maximum of MAX_FRAMES_FOR_SETUP_LIMIT frames from the video generator.

    Args:
        player_detection_model: The loaded YOLO player detection model.
        source_video_path: Path to the input video file.

    Returns:
        An initialized and fitted TeamClassifier instance, or None if setup failed or
        the classifier is unavailable.
    """
    if not TEAM_CLASSIFIER_AVAILABLE or TeamClassifier is None:
        logging.warning("TeamClassifier is not available or failed to import. Skipping setup.")
        return None

    # Define the maximum number of frames to process from the generator for setup
    # This could also be moved to your config.py file
    MAX_FRAMES_FOR_SETUP_LIMIT = 500

    logging.info("Starting Team Classifier setup...")
    logging.info(f"Collecting initial player crops from '{source_video_path}' with stride {config.TEAM_CLASSIFIER_STRIDE}.")
    logging.info(f"Will process a maximum of {MAX_FRAMES_FOR_SETUP_LIMIT} frames from the generator for setup.")

    # --- Collect Initial Player Crops ---
    crops = []
    frame_generator = None
    try:
        frame_generator = sv.get_video_frames_generator(
            source_path=source_video_path, stride=config.TEAM_CLASSIFIER_STRIDE
        )

        # Determine the total for tqdm progress bar
        tqdm_display_total = MAX_FRAMES_FOR_SETUP_LIMIT # Default to the max limit
        try:
            video_info = sv.VideoInfo.from_video_path(source_video_path)
            if video_info.total_frames and video_info.total_frames > 0:
                # Calculate how many frames the generator would yield without our new limit
                frames_generator_would_yield = video_info.total_frames // config.TEAM_CLASSIFIER_STRIDE
                # The tqdm total should be the lesser of what the generator would yield and our hard limit
                tqdm_display_total = min(frames_generator_would_yield, MAX_FRAMES_FOR_SETUP_LIMIT)
            # If video_info.total_frames is 0 or not available, tqdm_display_total remains MAX_FRAMES_FOR_SETUP_LIMIT
            elif video_info.total_frames == 0:
                tqdm_display_total = 0 # Video has no frames
                logging.warning("Video source reports 0 total frames.")

        except Exception as e:
            # If sv.VideoInfo fails, tqdm_display_total remains MAX_FRAMES_FOR_SETUP_LIMIT
            logging.warning(f"Could not determine video total frames for progress bar: {e}. Using setup limit for display.")

        processed_frames_count = 0
        with tqdm(frame_generator, desc="Collecting crops for TeamClassifier", total=tqdm_display_total, unit="frame") as pbar:
            for frame in pbar:
                if processed_frames_count >= MAX_FRAMES_FOR_SETUP_LIMIT:
                    logging.info(f"Reached frame processing limit for setup ({MAX_FRAMES_FOR_SETUP_LIMIT} frames). Stopping crop collection.")
                    break # Exit the loop as we've processed enough frames

                if frame is None:
                    logging.warning("Encountered None frame during initial crop collection.")
                    continue

                # Perform detection using the provided model
                results = player_detection_model.predict(frame, conf=config.DETECTION_CONFIDENCE_THRESHOLD, device=config.DEVICE, verbose=False)
                if not results or len(results) == 0:
                    processed_frames_count += 1 # Count this frame as processed even if no detections
                    continue

                detections = sv.Detections.from_ultralytics(results[0])
                # Apply NMS if needed (optional for setup, but good practice)
                detections = detections.with_nms(threshold=config.DETECTION_NMS_THRESHOLD, class_agnostic=True)

                # Filter for player detections using the NEW PLAYER_ID from config
                players_detections = detections[detections.class_id == config.PLAYER_ID]

                # Crop player images
                for xyxy in players_detections.xyxy:
                    crop = sv.crop_image(frame, xyxy)
                    if crop is not None and crop.size > 0:
                        # Ensure crop is in BGR format if needed by classifier
                        # (sv.crop_image returns BGR by default)
                        crops.append(crop)
                
                processed_frames_count += 1 # Increment after successfully processing a frame
                pbar.set_postfix({"crops_collected": len(crops), "frames_processed": processed_frames_count})


    except Exception as e:
        logging.error(f"Error during initial crop collection: {e}", exc_info=True)
        return None # Cannot proceed without crops
    finally:
        # Ensure generator is closed if it was opened
        if frame_generator and hasattr(frame_generator, 'close'):
            frame_generator.close()
            logging.info("Video frame generator closed.")


    if not crops:
        logging.error(f"No player crops were collected after processing {processed_frames_count} frames. Cannot fit TeamClassifier.")
        return None

    logging.info(f"Collected {len(crops)} player crops from {processed_frames_count} processed frames.")

    # --- Initialize and Fit TeamClassifier ---
    try:
        logging.info(f"Initializing TeamClassifier on device: {config.DEVICE}")
        # Pass the device specified in config
        team_classifier = TeamClassifier(device=str(config.DEVICE)) # Ensure device is passed as string if needed

        logging.info("Fitting TeamClassifier...")
        # The .fit() method expects a list of images (NumPy arrays)
        team_classifier.fit(crops) # Pass the collected BGR crops

        logging.info("TeamClassifier fitted successfully.")
        return team_classifier

    except Exception as e:
        logging.error(f"Error initializing or fitting TeamClassifier: {e}", exc_info=True)
        return None
