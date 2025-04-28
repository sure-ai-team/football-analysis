from ultralytics import YOLO
import supervision as sv
import numpy as np
import logging
from app.core import config # Import config to access constants

# Configure logging
logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Handles loading the YOLO model and performing object detection.
    """
    def __init__(self, model_path=config.YOLO_MODEL_PATH):
        """
        Initializes the ObjectDetector.

        Args:
            model_path (str or Path): Path to the YOLO model weights file.
        """
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Loads the YOLO model."""
        if not config.YOLO_MODEL_PATH.exists():
            logger.error(f"YOLO model not found at: {config.YOLO_MODEL_PATH}")
            raise FileNotFoundError(f"YOLO model not found at: {config.YOLO_MODEL_PATH}")
        try:
            model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded successfully from: {self.model_path}")
            # Perform a dummy inference to potentially speed up the first real one
            # model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {self.model_path}: {e}", exc_info=True)
            raise

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Performs object detection on a single frame.

        Args:
            frame: The input image frame (NumPy array).

        Returns:
            A supervision Detections object containing the detected objects.
            Returns sv.Detections.empty() if inference fails or no objects detected.
        """
        if self.model is None:
            logger.error("YOLO model is not loaded. Cannot predict.")
            return sv.Detections.empty()

        try:
            # Perform inference using the loaded YOLO model
            results = self.model.predict(
                frame,
                conf=config.YOLO_CONFIDENCE_THRESHOLD,
                iou=config.NMS_THRESHOLD, # Use NMS threshold from config
                device=config.DEVICE,
                verbose=False # Keep predictions quiet
            )

            # Check if results were returned and are not empty
            if not results or len(results) == 0:
                # logger.debug("No detections found in the frame.")
                return sv.Detections.empty()

            # Convert the first result (assuming single image inference) to sv.Detections
            detections = sv.Detections.from_ultralytics(results[0])
            # logger.debug(f"Detected {len(detections)} objects.")
            return detections

        except Exception as e:
            logger.error(f"Error during YOLO prediction: {e}", exc_info=True)
            return sv.Detections.empty()

# Example usage (optional, for testing)
if __name__ == "__main__":
    import cv2
    from app.core import config

    # Ensure the test video path is correct
    if not config.DEFAULT_SOURCE_VIDEO_PATH.exists():
        print(f"Test video not found at {config.DEFAULT_SOURCE_VIDEO_PATH}. Skipping detection test.")
    else:
        print("Initializing detector...")
        try:
            detector = ObjectDetector()
            print("Detector initialized.")

            print(f"Loading frame from {config.DEFAULT_SOURCE_VIDEO_PATH}...")
            cap = cv2.VideoCapture(str(config.DEFAULT_SOURCE_VIDEO_PATH))
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print("Frame loaded. Performing prediction...")
                    detections = detector.predict(frame)
                    print(f"Prediction complete. Found {len(detections)} detections.")

                    # Example: Print details of first 5 detections
                    for i in range(min(5, len(detections))):
                        print(f"  Detection {i}: xyxy={detections.xyxy[i]}, "
                              f"conf={detections.confidence[i]:.2f}, "
                              f"class_id={detections.class_id[i]}")

                    # Optional: Visualize detections
                    # box_annotator = sv.BoxAnnotator()
                    # annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                    # cv2.imshow("Detections", annotated_frame)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                else:
                    print("Failed to read frame from video.")
                cap.release()
            else:
                print("Failed to open video capture.")
        except Exception as e:
            print(f"An error occurred during the detection test: {e}")

