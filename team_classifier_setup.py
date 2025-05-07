import supervision as sv
import numpy as np
from tqdm import tqdm
import logging
import config
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

# Example usage (optional, for testing)
if __name__ == "__main__":
    print("Testing Team Classifier Setup...")
    # This requires a valid video path and a loaded detection model
    # Replace with actual paths/models for testing
    # Ensure your config.py has appropriate values for:
    # TEAM_CLASSIFIER_STRIDE, DETECTION_CONFIDENCE_THRESHOLD, DEVICE, PLAYER_ID
    
    # Create a dummy config.py if it doesn't exist for basic testing structure
    if not os.path.exists("config.py"):
        with open("config.py", "w") as f:
            f.write("TEAM_CLASSIFIER_STRIDE = 10\n")
            f.write("DETECTION_CONFIDENCE_THRESHOLD = 0.3\n")
            f.write("DEVICE = 'cpu'\n") # or 'cuda' if available
            f.write("PLAYER_ID = 0 # Assuming player class ID is 0\n")
            f.write("DETECTION_NMS_THRESHOLD = 0.5\n") # Example, if you uncomment NMS
        print("Created a dummy config.py for testing.")

    # Configure logging for the test
    logging.basicConfig(level=logging.INFO)


    TEST_VIDEO = "path/to/your/test_video.mp4" # <--- !!! REPLACE WITH A VALID VIDEO PATH !!!
    
    # Mock player_detection_model and its predict method for standalone testing
    class MockPlayerDetectionModel:
        def predict(self, frame, conf, device, verbose):
            # Simulate some detections
            # In a real scenario, this would return actual detection results
            # For testing, let's assume it finds one player in the center
            h, w, _ = frame.shape
            # Return a list containing one result object (like Ultralytics output)
            class MockResult:
                def __init__(self, boxes_data, orig_img_shape):
                    self.boxes = MockBoxes(boxes_data) # ultralytics.engine.results.Boxes
                    self.orig_shape = orig_img_shape # e.g., (720, 1280)

            class MockBoxes: # Simulates ultralytics.engine.results.Boxes
                def __init__(self, data):
                    # data is expected to be a tensor or ndarray like [x1, y1, x2, y2, conf, cls_id]
                    self.data = np.array(data) if data else np.empty((0,6))
                @property
                def xyxy(self): return self.data[:, :4]
                @property
                def conf(self): return self.data[:, 4]
                @property
                def cls(self): return self.data[:, 5]


            # Simulate one detection of class PLAYER_ID (e.g., 0)
            player_box = [w*0.4, h*0.4, w*0.6, h*0.6, 0.9, config.PLAYER_ID] # x1,y1,x2,y2,conf,class_id
            
            # Simulate no detections sometimes
            if np.random.rand() < 0.3: # 30% chance of no detections
                 mock_result_data = []
            else:
                 mock_result_data = [player_box]

            return [MockResult(mock_result_data, frame.shape[:2])]


    # Attempt to load a real model if available, otherwise use mock
    real_detector = None
    try:
        from models import load_player_detection_model # Assuming this exists
        # Check if a model can be loaded (e.g., by checking a path or a flag)
        # For this example, let's assume it tries to load and might fail
        # real_detector = load_player_detection_model() # This would be your actual model loading
        pass # Keep real_detector as None to use Mock by default for this example
    except ImportError:
        print("Skipping real model loading: 'models.py' or 'load_player_detection_model' not found.")
    
    test_detector_to_use = real_detector if real_detector else MockPlayerDetectionModel()
    print(f"Using {'real' if real_detector else 'mock'} player detection model for the test.")

    if not os.path.exists(TEST_VIDEO):
        print(f"Test video not found at '{TEST_VIDEO}'. Please update the path.")
        # Create a dummy video file for testing if it doesn't exist
        # This requires OpenCV to be installed (cv2)
        try:
            print(f"Attempting to create a dummy video file: {TEST_VIDEO}")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # Create a short video (e.g., 600 frames to test the 500 limit)
            num_dummy_frames = 700 # More than MAX_FRAMES_FOR_SETUP_LIMIT
            out = cv2.VideoWriter(TEST_VIDEO, fourcc, 20.0, (640, 480))
            for _ in range(num_dummy_frames):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, 'Dummy Frame', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                out.write(frame)
            out.release()
            print(f"Dummy video '{TEST_VIDEO}' created with {num_dummy_frames} frames. Please re-run the test.")
        except Exception as e:
            print(f"Could not create dummy video: {e}. Please provide a valid TEST_VIDEO path.")
    else:
        print(f"Test video found at '{TEST_VIDEO}'.")
        if test_detector_to_use:
            print("Proceeding with Team Classifier setup test...")
            classifier = setup_team_classifier(test_detector_to_use, TEST_VIDEO)
            if classifier:
                print("Team Classifier setup test completed successfully.")
            else:
                print("Team Classifier setup test failed.")
        else:
            print("Skipping test: Detection model (real or mock) is not available.")

