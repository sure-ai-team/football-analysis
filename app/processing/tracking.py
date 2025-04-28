import numpy as np
import supervision as sv
from boxmot import BotSort
import logging
import torch # To check for exceptions

from app.core import config

# Configure logging
logger = logging.getLogger(__name__)

class ObjectTracker:
    """
    Handles initializing and updating the object tracker (BoTSORT).
    """
    def __init__(self):
        """Initializes the BotSort tracker."""
        self.tracker = self._initialize_tracker()

    def _initialize_tracker(self):
        """Loads and configures the BotSort tracker."""
        try:
            # Check if ReID weights exist if specified in config
            reid_weights_path = config.REID_WEIGHTS_PATH if config.TRACKER_WITH_REID else None
            if config.TRACKER_WITH_REID and not config.REID_WEIGHTS_PATH.exists():
                logger.warning(f"ReID weights specified but not found at {config.REID_WEIGHTS_PATH}. Disabling ReID.")
                reid_weights_path = None # Disable ReID if weights are missing

            tracker = BotSort(
                reid_weights=reid_weights_path,
                device=config.DEVICE,
                half=config.TRACKER_HALF_PRECISION,
                with_reid=(reid_weights_path is not None) # Set based on actual path used
            )
            logger.info(f"BoTSORT tracker initialized successfully on device '{config.DEVICE}' "
                        f"with ReID: {tracker.with_reid}.")
            return tracker
        except Exception as e:
            logger.error(f"Failed to initialize BoTSORT tracker: {e}", exc_info=True)
            return None # Return None if initialization fails

    def update(self, frame: np.ndarray, classified_detections: sv.Detections) -> sv.Detections:
        """
        Updates the tracker with the current frame's classified detections.

        Args:
            frame: The current video frame (NumPy array). Required by BoTSORT.
            classified_detections: sv.Detections object containing the detections
                                   classified by team/role (output from classification module).

        Returns:
            sv.Detections object with tracker IDs assigned. Returns empty Detections
            if tracking fails or no tracks are updated.
        """
        if self.tracker is None:
            logger.error("Tracker not initialized. Cannot update.")
            return sv.Detections.empty()

        if len(classified_detections) == 0:
            # Update tracker with empty input if no detections
            try:
                # BoTSORT expects input shape (N, 6): [x1, y1, x2, y2, conf, cls]
                empty_input = np.empty((0, 6))
                self.tracker.update(empty_input, frame)
                # logger.debug("Tracker updated with empty detections.")
            except (Exception, torch.cuda.OutOfMemoryError) as e: # Catch potential CUDA errors too
                logger.error(f"Error updating tracker with empty input: {e}", exc_info=True)
            return sv.Detections.empty() # Return empty as no tracks were updated

        # Prepare input for BoTSORT: [x1, y1, x2, y2, conf, cls]
        # Ensure confidence and class_id are present and correctly shaped
        if classified_detections.confidence is None:
             logger.warning("Detections missing confidence scores. Assigning default 1.0 for tracking.")
             confidences = np.ones(len(classified_detections))[:, np.newaxis]
        else:
             confidences = classified_detections.confidence[:, np.newaxis]

        if classified_detections.class_id is None:
             logger.error("Detections missing class_id. Cannot track.")
             return sv.Detections.empty()
        else:
             class_ids = classified_detections.class_id[:, np.newaxis]


        # Combine xyxy, confidence, and class_id
        boxmot_input = np.hstack((
            classified_detections.xyxy,
            confidences,
            class_ids
        ))

        try:
            # Update the tracker
            tracks = self.tracker.update(boxmot_input, frame)

            # Process tracker output
            if tracks.shape[0] > 0:
                # Output format: [x1, y1, x2, y2, track_id, conf, cls, ind]
                # We need: xyxy, tracker_id, confidence, class_id
                tracked_detections = sv.Detections(
                    xyxy=tracks[:, 0:4],
                    tracker_id=tracks[:, 4].astype(int),
                    confidence=tracks[:, 5],
                    class_id=tracks[:, 6].astype(int)
                )
                # logger.debug(f"Tracker updated. Found {len(tracked_detections)} active tracks.")
                return tracked_detections
            else:
                # logger.debug("Tracker updated, but no active tracks returned.")
                return sv.Detections.empty()

        except (Exception, torch.cuda.OutOfMemoryError) as e:
            logger.error(f"Error during tracker update: {e}", exc_info=True)
            # Attempt to clear CUDA cache if it's an OOM error
            if isinstance(e, torch.cuda.OutOfMemoryError):
                logger.warning("CUDA OOM error during tracking. Attempting to clear cache.")
                torch.cuda.empty_cache()
            return sv.Detections.empty() # Return empty on error

# Example usage (optional, for testing)
if __name__ == "__main__":
    print("Initializing tracker...")
    try:
        tracker = ObjectTracker()
        print(f"Tracker initialized: {'OK' if tracker.tracker else 'Failed'}")

        # Create dummy detections for testing
        dummy_detections = sv.Detections(
            xyxy=np.array([
                [100, 100, 200, 200],
                [300, 300, 400, 400]
            ]),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([config.TEAM_A_ID, config.TEAM_B_ID]) # Use valid team IDs
        )
        print(f"\nDummy detections created: {len(dummy_detections)}")

        # Create a dummy frame
        dummy_frame = np.zeros((600, 800, 3), dtype=np.uint8)
        print("Dummy frame created.")

        # Perform tracker update
        print("\nUpdating tracker...")
        tracked_dets = tracker.update(dummy_frame, dummy_detections)
        print(f"Tracker update complete. Result: {len(tracked_dets)} tracked detections.")

        if len(tracked_dets) > 0:
            print("Tracked Detections Details:")
            for i in range(len(tracked_dets)):
                 print(f"  Track {i}: xyxy={tracked_dets.xyxy[i]}, "
                       f"track_id={tracked_dets.tracker_id[i]}, "
                       f"conf={tracked_dets.confidence[i]:.2f}, "
                       f"class_id={tracked_dets.class_id[i]}")

        # Test update with empty detections
        print("\nUpdating tracker with empty detections...")
        tracked_dets_empty = tracker.update(dummy_frame, sv.Detections.empty())
        print(f"Tracker update complete. Result: {len(tracked_dets_empty)} tracked detections.")

    except Exception as e:
        print(f"An error occurred during the tracking test: {e}")

