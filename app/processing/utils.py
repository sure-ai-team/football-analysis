import numpy as np
import supervision as sv
import cv2
import logging

# Configure logging
logger = logging.getLogger(__name__)

def color_distance(color1: sv.Color | None, color2: sv.Color | None) -> float:
    """
    Calculates the Euclidean distance between two sv.Color objects in RGB space.

    Args:
        color1: The first supervision Color object (or None).
        color2: The second supervision Color object (or None).

    Returns:
        The Euclidean distance, or float('inf') if either color is None or invalid.
    """
    if color1 is None or color2 is None:
        return float('inf') # Return infinity if a color is missing

    # Ensure colors have r, g, b attributes
    if not all(hasattr(c, attr) for c in [color1, color2] for attr in ['r', 'g', 'b']):
          logger.warning(f"Invalid color object passed to color_distance: {color1}, {color2}")
          return float('inf')
    try:
        rgb1 = np.array([color1.r, color1.g, color1.b])
        rgb2 = np.array([color2.r, color2.g, color2.b])
        return np.linalg.norm(rgb1 - rgb2)
    except Exception as e:
        logger.error(f"Error calculating color distance: {e}")
        return float('inf')

def calculate_average_color(
    frame: np.ndarray,
    detections: sv.Detections,
    central_fraction: float = 0.5
) -> sv.Color | None:
    """
    Calculates the average color from the central region of detection boxes.

    Args:
        frame: The image frame (NumPy array BGR).
        detections: A supervision Detections object.
        central_fraction: The fraction of the width and height of the central
                          region to consider for color calculation.

    Returns:
        An sv.Color object representing the average color, or None if no valid
        detections or crops are found.
    """
    if len(detections) == 0:
        return None

    avg_colors_bgr = []
    frame_height, frame_width, _ = frame.shape

    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)

        # Clamp coordinates to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)

        # Skip invalid boxes
        if x1 >= x2 or y1 >= y2:
            continue

        # Calculate central region coordinates
        box_w, box_h = x2 - x1, y2 - y1
        center_x, center_y = x1 + box_w // 2, y1 + box_h // 2
        central_w, central_h = int(box_w * central_fraction), int(box_h * central_fraction)

        cx1 = max(x1, center_x - central_w // 2)
        cy1 = max(y1, center_y - central_h // 2)
        cx2 = min(x2, center_x + central_w // 2)
        cy2 = min(y2, center_y + central_h // 2)

        # Skip invalid central regions
        if cx1 >= cx2 or cy1 >= cy2:
            continue

        # Crop the central region
        crop = frame[cy1:cy2, cx1:cx2]

        # Calculate average color if crop is valid
        if crop.size > 0:
            avg_bgr = cv2.mean(crop)[:3] # Get B, G, R components
            avg_colors_bgr.append(avg_bgr)
        # else:
            # logger.debug(f"Skipping empty crop for box {xyxy}")

    if not avg_colors_bgr:
        # logger.warning("No valid crops found to calculate average color.")
        return None

    # Calculate the final average BGR across all valid crops
    final_avg_bgr = np.mean(avg_colors_bgr, axis=0)
    b, g, r = map(int, final_avg_bgr)

    # Ensure minimum intensity to avoid pure black if desired (optional)
    # min_intensity = 50
    # if r < min_intensity and g < min_intensity and b < min_intensity:
    #     r, g, b = min_intensity, min_intensity, min_intensity

    return sv.Color(r=r, g=g, b=b)

def get_video_info(video_path: str) -> sv.VideoInfo | None:
    """Safely gets video info using supervision."""
    try:
        return sv.VideoInfo.from_video_path(video_path)
    except Exception as e:
        logger.warning(f"Could not get video info using supervision for {video_path}. Error: {e}")
        # Fallback using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file with OpenCV: {video_path}")
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not fps or fps <= 0: fps = 30 # Default FPS if detection fails
        if frame_count <= 0: frame_count = None # Use None if count is invalid
        logger.warning(f"Using OpenCV fallback for video info: {width}x{height}, FPS: {fps}, Frames: {frame_count}")
        # Create a basic VideoInfo-like object (or return dict if preferred)
        class FallbackVideoInfo:
            def __init__(self, w, h, f, fc):
                self.width = w
                self.height = h
                self.fps = f
                self.total_frames = fc
        return FallbackVideoInfo(width, height, fps, frame_count)

