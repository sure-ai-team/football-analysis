# team_classifier_setup.py
"""
Handles the setup for the TeamClassifier, including collecting
initial player crops and fitting the classifier.
"""
import supervision as sv
import numpy as np
from tqdm import tqdm
import logging
import config
import cv2

# Attempt to import the TeamClassifier - replace with actual import path
try:
    # This is a placeholder - replace 'sports.common.team' with the actual module
    # where your TeamClassifier is defined. If it's in the same project,
    # you might use something like 'from .team_classifier_module import TeamClassifier'
    from sports.common.team import TeamClassifier # <--- ADJUST THIS IMPORT
    TEAM_CLASSIFIER_AVAILABLE = True
except ImportError:
    logging.error("Failed to import TeamClassifier from 'sports.common.team'. Please ensure it's installed and the path is correct. Team classification will be disabled.")
    TeamClassifier = None # Define as None if import fails
    TEAM_CLASSIFIER_AVAILABLE = False


def setup_team_classifier(player_detection_model, source_video_path: str):
    """
    Collects player crops from the initial part of the video and fits the TeamClassifier.

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

    logging.info("Starting Team Classifier setup...")
    logging.info(f"Collecting initial player crops from '{source_video_path}' with stride {config.TEAM_CLASSIFIER_STRIDE}")

    # --- Collect Initial Player Crops ---
    crops = []
    frame_generator = None
    try:
        frame_generator = sv.get_video_frames_generator(
            source_path=source_video_path, stride=config.TEAM_CLASSIFIER_STRIDE
        )

        # Estimate total frames for progress bar (optional but helpful)
        try:
             video_info = sv.VideoInfo.from_video_path(source_video_path)
             total_setup_frames = (video_info.total_frames // config.TEAM_CLASSIFIER_STRIDE) if video_info.total_frames else None
        except Exception:
             total_setup_frames = None # Cannot determine total

        with tqdm(frame_generator, desc="Collecting crops for TeamClassifier", total=total_setup_frames, unit="frame") as pbar:
            for frame in pbar:
                if frame is None:
                    logging.warning("Encountered None frame during initial crop collection.")
                    continue

                # Perform detection using the provided model
                results = player_detection_model.predict(frame, conf=config.DETECTION_CONFIDENCE_THRESHOLD, device=config.DEVICE, verbose=False)
                if not results or len(results) == 0:
                    continue

                detections = sv.Detections.from_ultralytics(results[0])
                # Apply NMS if needed (optional for setup, but good practice)
                # detections = detections.with_nms(threshold=config.DETECTION_NMS_THRESHOLD, class_agnostic=True)

                # Filter for player detections
                players_detections = detections[detections.class_id == config.PLAYER_ID]

                # Crop player images
                for xyxy in players_detections.xyxy:
                    crop = sv.crop_image(frame, xyxy)
                    if crop is not None and crop.size > 0:
                        # Ensure crop is in BGR format if needed by classifier
                        # (sv.crop_image returns BGR by default)
                        crops.append(crop)
                pbar.set_postfix({"crops_collected": len(crops)})


    except Exception as e:
        logging.error(f"Error during initial crop collection: {e}", exc_info=True)
        return None # Cannot proceed without crops
    finally:
        # Ensure generator is closed if it was opened
        if frame_generator and hasattr(frame_generator, 'close'):
            frame_generator.close()


    if not crops:
        logging.error("No player crops were collected. Cannot fit TeamClassifier.")
        return None

    logging.info(f"Collected {len(crops)} player crops.")

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

# Example usage (optional, for testing)
if __name__ == "__main__":
    print("Testing Team Classifier Setup...")
    # This requires a valid video path and a loaded detection model
    # Replace with actual paths/models for testing
    TEST_VIDEO = "path/to/your/test_video.mp4"
    # Assuming models.py is in the same directory and load_player_detection_model works
    try:
        from models import load_player_detection_model
        test_detector = load_player_detection_model()
        if test_detector and os.path.exists(TEST_VIDEO):
             classifier = setup_team_classifier(test_detector, TEST_VIDEO)
             if classifier:
                 print("Team Classifier setup test completed successfully.")
             else:
                 print("Team Classifier setup test failed.")
        else:
             print("Skipping test: Need a valid video path and detection model.")
    except ImportError:
        print("Skipping test: Could not import model loader.")
    except Exception as e:
        print(f"Error during test: {e}")

