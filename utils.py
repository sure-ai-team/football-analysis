# utils.py
"""
Utility functions for color calculations, OCR processing, and goalkeeper resolution.
"""
import numpy as np
import supervision as sv
import cv2
import config
import logging
import os

# --- Color Utilities ---

def calculate_average_color(frame: np.ndarray, detections: sv.Detections, central_fraction: float = config.CENTRAL_FRACTION_FOR_COLOR) -> sv.Color | None:
    """
    Calculates the average color from the central region of detection boxes.

    Args:
        frame: The input image frame (NumPy array BGR).
        detections: A supervision.Detections object.
        central_fraction: The fraction of the center area of the bounding box to use.

    Returns:
        An sv.Color object representing the average color, or None if no valid detections
        or crops could be processed.
    """
    if len(detections) == 0:
        return None

    avg_colors = []
    height, width, _ = frame.shape

    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        # Clamp coordinates to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        if x1 >= x2 or y1 >= y2:
            continue # Skip invalid boxes

        box_w, box_h = x2 - x1, y2 - y1
        center_x, center_y = x1 + box_w // 2, y1 + box_h // 2

        # Calculate central region coordinates
        central_w, central_h = int(box_w * central_fraction), int(box_h * central_fraction)
        cx1 = max(x1, center_x - central_w // 2)
        cy1 = max(y1, center_y - central_h // 2)
        cx2 = min(x2, center_x + central_w // 2)
        cy2 = min(y2, center_y + central_h // 2)

        if cx1 >= cx2 or cy1 >= cy2:
            continue # Skip if central region is invalid

        # Crop the central region
        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size > 0:
            # Calculate the mean color (BGR)
            avg_bgr = cv2.mean(crop)[:3]
            avg_colors.append(avg_bgr)

    if not avg_colors:
        return None

    # Calculate the final average BGR color across all valid detections
    final_avg_bgr = np.mean(avg_colors, axis=0)
    b, g, r = map(int, final_avg_bgr)

    # Ensure minimum intensity to avoid pure black if needed (optional)
    min_intensity = 50
    if r < min_intensity and g < min_intensity and b < min_intensity:
        logging.debug("Calculated average color was very dark, adjusting to minimum intensity.")
        r, g, b = min_intensity, min_intensity, min_intensity

    return sv.Color(r=r, g=g, b=b)


def color_distance(color1: sv.Color | None, color2: sv.Color | None) -> float:
    """
    Calculates Euclidean distance between two sv.Color objects in RGB space.

    Args:
        color1: The first sv.Color object.
        color2: The second sv.Color object.

    Returns:
        The Euclidean distance, or float('inf') if either color is None or invalid.
    """
    if color1 is None or color2 is None:
        return float('inf') # Return infinity if a color is missing

    # Ensure colors have r, g, b attributes
    if not all(hasattr(c, attr) for c in [color1, color2] for attr in ['r', 'g', 'b']):
         logging.warning(f"Invalid color object passed to color_distance: {color1}, {color2}")
         return float('inf')

    try:
        # Extract RGB values and calculate distance
        rgb1 = np.array([color1.r, color1.g, color1.b])
        rgb2 = np.array([color2.r, color2.g, color2.b])
        distance = np.linalg.norm(rgb1 - rgb2)
        return distance
    except Exception as e:
        logging.error(f"Error calculating color distance: {e}", exc_info=True)
        return float('inf')

# --- Goalkeeper Resolution ---

def resolve_goalkeepers_team_id(
    frame: np.ndarray,
    goalkeepers: sv.Detections,
    team_a_color: sv.Color | None,
    team_b_color: sv.Color | None,
    color_similarity_threshold: float = config.COLOR_SIMILARITY_THRESHOLD
) -> np.ndarray:
    """
    Assigns team IDs to goalkeepers based primarily on color similarity to team average colors,
    with a positional fallback for ambiguous cases.

    Args:
        frame: The input image frame (NumPy array BGR).
        goalkeepers: A supervision.Detections object containing only goalkeepers.
        team_a_color: The calculated average sv.Color for Team A.
        team_b_color: The calculated average sv.Color for Team B.
        color_similarity_threshold: The maximum RGB distance difference to consider colors ambiguous.

    Returns:
        A NumPy array of assigned team IDs (config.TEAM_A_ID or config.TEAM_B_ID)
        for each goalkeeper, or -1 if assignment failed for a specific GK.
    """
    goalkeeper_team_ids = []
    if len(goalkeepers) == 0:
        return np.array([], dtype=int) # Return empty if no goalkeepers

    frame_height, frame_width, _ = frame.shape

    # Check if we have valid team colors to compare against
    valid_team_colors = team_a_color is not None and team_b_color is not None

    for i in range(len(goalkeepers)):
        # Process each goalkeeper individually
        gk_detection_single = goalkeepers[i:i+1] # Create Detections object with one GK
        gk_center_x, _ = gk_detection_single.get_anchors_coordinates(sv.Position.CENTER)[0]

        assigned_id = -1 # Default to invalid ID

        # 1. Try Color Similarity if valid team colors are available
        if valid_team_colors:
            gk_color = calculate_average_color(frame, gk_detection_single)

            if gk_color is not None:
                dist_a = color_distance(gk_color, team_a_color)
                dist_b = color_distance(gk_color, team_b_color)

                # Check if the difference in distances is significant enough
                if abs(dist_a - dist_b) > color_similarity_threshold:
                    assigned_id = config.TEAM_A_ID if dist_a < dist_b else config.TEAM_B_ID
                    logging.debug(f"GK {i} assigned by color: Team {'A' if assigned_id == config.TEAM_A_ID else 'B'} (Dist A: {dist_a:.1f}, Dist B: {dist_b:.1f})")
                else:
                    # Colors are too similar, will proceed to positional fallback
                    logging.debug(f"GK {i} color ambiguous (Dist A: {dist_a:.1f}, Dist B: {dist_b:.1f}). Using fallback.")
            else:
                # Failed to calculate GK color, will proceed to positional fallback
                logging.debug(f"GK {i} color calculation failed. Using fallback.")
        else:
             logging.debug(f"GK {i} team colors invalid. Using fallback.")


        # 2. Positional Fallback (if color failed, was ambiguous, or team colors were invalid)
        if assigned_id == -1:
            logging.debug(f"GK {i} using positional fallback.")
            # Assign based on which half of the pitch they are on
            # Assumes Team A (ID 0) defends left goal, Team B (ID 1) defends right
            assigned_id = config.TEAM_A_ID if gk_center_x < frame_width / 2 else config.TEAM_B_ID
            logging.debug(f"GK {i} assigned by position: Team {'A' if assigned_id == config.TEAM_A_ID else 'B'} (Center X: {gk_center_x:.0f} / Frame Width: {frame_width})")


        goalkeeper_team_ids.append(assigned_id)

    return np.array(goalkeeper_team_ids, dtype=int)


# --- OCR Utilities ---

def perform_ocr_on_crop(ocr_model, crop: np.ndarray, ocr_available: bool) -> tuple[str | None, float | None]:
    """
    Performs OCR on a given crop using the loaded PaddleOCR model.

    Args:
        ocr_model: The initialized PaddleOCR model object (or None).
        crop: The image crop (NumPy array BGR or Grayscale).
        ocr_available: Boolean flag indicating if OCR model was loaded successfully.

    Returns:
        A tuple containing:
        - The best detected digit sequence (string) or None.
        - The confidence score (float) for the best sequence or None.
    """
    if not ocr_available or ocr_model is None or crop is None or crop.size == 0:
        return None, None

    try:
        # PaddleOCR expects BGR format
        if len(crop.shape) == 2: # If grayscale, convert back to BGR for OCR
             crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        else:
             crop_bgr = crop

        result = ocr_model.ocr(crop_bgr, cls=config.PADDLEOCR_USE_ANGLE_CLS)

        best_num_str: str | None = None
        highest_conf: float = 0.0

        # PaddleOCR result structure can vary slightly; handle potential nesting
        if result and isinstance(result, list) and len(result) > 0:
            # Sometimes results are nested [[...]]
            ocr_data = result[0] if isinstance(result[0], list) else result

            if ocr_data: # Check if the inner list is not empty
                for res_item in ocr_data:
                    # Ensure the item structure is as expected: [bbox, (text, confidence)]
                    if (isinstance(res_item, list) and len(res_item) == 2 and
                        isinstance(res_item[1], (tuple, list)) and len(res_item[1]) == 2):

                        text, confidence = res_item[1]

                        # Validate text and confidence
                        if (isinstance(text, str) and text.isdigit() and
                            config.MIN_JERSEY_DIGITS <= len(text) <= config.MAX_JERSEY_DIGITS and
                            isinstance(confidence, (float, int)) and # Allow int confidence too
                            confidence > config.OCR_CONFIDENCE_THRESHOLD):

                            # Check if this is the best confidence digit sequence found so far
                            if float(confidence) > highest_conf:
                                highest_conf = float(confidence)
                                best_num_str = text
                        # else: logging.debug(f"OCR result discarded: Text='{text}', Conf={confidence}") # Optional debug
                    # else: logging.debug(f"Unexpected OCR item structure: {res_item}") # Optional debug

        # Return the best number found (or None) and its confidence
        return best_num_str, highest_conf if best_num_str else None

    except Exception as e:
        logging.error(f"Error during PaddleOCR inference: {e}", exc_info=True)
        return None, None

def save_ocr_debug_crop(frame_idx: int, track_id: int, player_crop: np.ndarray, ocr_input_crop: np.ndarray):
    """Saves crops used for OCR for debugging purposes."""
    try:
        if not os.path.exists(config.OCR_DEBUG_DIR):
            os.makedirs(config.OCR_DEBUG_DIR)
            logging.info(f"Created OCR debug directory: {config.OCR_DEBUG_DIR}")

        player_filename = os.path.join(config.OCR_DEBUG_DIR, f"frame{frame_idx}_track{track_id}_player.png")
        ocr_input_filename = os.path.join(config.OCR_DEBUG_DIR, f"frame{frame_idx}_track{track_id}_ocr_input.png")

        cv2.imwrite(player_filename, player_crop)
        cv2.imwrite(ocr_input_filename, ocr_input_crop)
        # logging.debug(f"Saved OCR debug crops for frame {frame_idx}, track {track_id}")
    except Exception as write_e:
        logging.warning(f"Error saving OCR debug crop for frame {frame_idx}, track {track_id}: {write_e}")

