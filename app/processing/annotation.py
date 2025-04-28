import numpy as np
import supervision as sv
import cv2
import random
from collections import deque
import logging

from app.core import config

# Configure logging
logger = logging.getLogger(__name__)

# --- Annotator Initialization ---
# We will create annotators dynamically based on team colors per frame,
# but we can define styles here.

# Define default annotators (colors will be overridden)
DEFAULT_ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=config.FALLBACK_COLOR,
    thickness=config.ELLIPSE_THICKNESS
)
DEFAULT_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=config.FALLBACK_COLOR,
    text_color=config.LABEL_TEXT_COLOR,
    text_position=config.LABEL_TEXT_POSITION,
    text_scale=config.LABEL_TEXT_SCALE,
    text_thickness=config.LABEL_TEXT_THICKNESS
)

def draw_ball_trail(
    frame: np.ndarray,
    ball_positions: deque # Deque containing (x, y) tuples
    ) -> np.ndarray:
    """
    Draws a 'magical' trail for the ball's recent positions.

    Args:
        frame: The frame to draw on (BGR NumPy array).
        ball_positions: A deque holding the recent center coordinates of the ball.

    Returns:
        The frame with the ball trail annotated.
    """
    annotated_frame = frame.copy()
    num_points = len(ball_positions)

    if num_points < 2:
        return annotated_frame # Need at least two points for a line

    try:
        # Draw lines between consecutive points
        for i in range(1, num_points):
            pt1 = ball_positions[i-1]
            pt2 = ball_positions[i]

            # Ensure points are valid tuples
            if not (isinstance(pt1, tuple) and len(pt1) == 2 and
                    isinstance(pt2, tuple) and len(pt2) == 2):
                # logger.warning(f"Invalid point format in ball_positions: pt1={pt1}, pt2={pt2}")
                continue # Skip this segment

            # Draw the main trail line
            cv2.line(annotated_frame, pt1, pt2, config.BALL_TRAIL_BASE_COLOR, config.BALL_TRAIL_THICKNESS)

            # --- Add Sparkle Effect ---
            # Calculate intensity based on position in the trail (newer points brighter)
            # Use (i / num_points) for fraction, avoids division by zero if num_points=1 (though checked earlier)
            alpha_fraction = i / float(num_points)
            sparkle_intensity = int(config.SPARKLE_BASE_INTENSITY + (config.SPARKLE_MAX_INTENSITY - config.SPARKLE_BASE_INTENSITY) * alpha_fraction)
            sparkle_intensity = max(0, min(255, sparkle_intensity)) # Clamp intensity
            sparkle_color = (sparkle_intensity, sparkle_intensity, sparkle_intensity) # Grayscale sparkle

            # Draw multiple sparkles around the end point (pt2)
            for _ in range(config.SPARKLE_COUNT):
                offset_x = random.randint(-config.SPARKLE_OFFSET, config.SPARKLE_OFFSET)
                offset_y = random.randint(-config.SPARKLE_OFFSET, config.SPARKLE_OFFSET)
                sparkle_pt = (pt2[0] + offset_x, pt2[1] + offset_y)

                # Draw the sparkle
                cv2.circle(
                    annotated_frame,
                    sparkle_pt,
                    config.SPARKLE_RADIUS,
                    sparkle_color,
                    -1 # Filled circle
                )
    except Exception as e:
         logger.error(f"Error drawing ball trail: {e}", exc_info=True)
         return frame # Return original frame on error

    return annotated_frame


def draw_current_ball_marker(frame: np.ndarray, ball_positions: deque) -> np.ndarray:
    """Draws a marker at the most recent ball position."""
    annotated_frame = frame.copy()
    if ball_positions:
        last_pos = ball_positions[-1]
        if isinstance(last_pos, tuple) and len(last_pos) == 2:
            try:
                cv2.circle(
                    annotated_frame,
                    last_pos,
                    config.CURRENT_BALL_MARKER_RADIUS,
                    config.CURRENT_BALL_MARKER_COLOR,
                    config.CURRENT_BALL_MARKER_THICKNESS # Filled
                )
            except Exception as e:
                 logger.error(f"Error drawing current ball marker: {e}", exc_info=True)
                 return frame # Return original frame on error
    return annotated_frame


def annotate_frame(
    frame: np.ndarray,
    tracked_detections: sv.Detections,
    final_labels: list[str],
    dynamic_color_map: dict[int, sv.Color],
    ball_positions: deque | None
    ) -> np.ndarray:
    """
    Annotates a frame with tracked detections (ellipses, labels) and ball trail.

    Args:
        frame: The original frame (BGR NumPy array).
        tracked_detections: sv.Detections object with tracker_id and classified class_id.
        final_labels: A list of strings corresponding to the labels for each tracked detection.
        dynamic_color_map: Dictionary mapping team/role IDs to sv.Color for this frame.
        ball_positions: Deque of recent ball center coordinates.

    Returns:
        The annotated frame (BGR NumPy array).
    """
    annotated_frame = frame.copy()

    # --- Annotate Ball Trail and Marker ---
    if ball_positions is not None:
        annotated_frame = draw_ball_trail(annotated_frame, ball_positions)
        annotated_frame = draw_current_ball_marker(annotated_frame, ball_positions)

    # --- Annotate Tracked People ---
    if len(tracked_detections) == 0 or len(tracked_detections) != len(final_labels):
        if len(tracked_detections) != len(final_labels):
             logger.warning(f"Mismatch between tracked detections ({len(tracked_detections)}) and labels ({len(final_labels)}). Skipping people annotation.")
        # else: logger.debug("No tracked people to annotate.")
        return annotated_frame # Return frame with only ball trail (if any)

    # Get unique team/role IDs present in the current tracked detections
    try:
        unique_team_ids = np.unique(tracked_detections.class_id)
    except Exception as e:
        logger.error(f"Could not get unique team IDs from tracked detections: {e}. Skipping annotation.")
        return annotated_frame


    for current_team_id in unique_team_ids:
        # Filter detections and labels for the current team/role
        team_mask = (tracked_detections.class_id == current_team_id)
        team_detections = tracked_detections[team_mask]
        # Ensure labels are filtered correctly using the boolean mask
        team_labels = [label for i, label in enumerate(final_labels) if team_mask[i]]


        if len(team_detections) == 0:
            continue # Should not happen if unique_team_ids is correct, but check anyway

        # Get the appropriate color for this team/role
        team_color = dynamic_color_map.get(current_team_id, config.FALLBACK_COLOR)

        # Create temporary annotators with the correct color
        # This is safer than modifying the default annotators
        temp_ellipse_annotator = sv.EllipseAnnotator(
            color=team_color,
            thickness=config.ELLIPSE_THICKNESS
        )
        temp_label_annotator = sv.LabelAnnotator(
            color=team_color, # Background color of the label box
            text_color=config.LABEL_TEXT_COLOR,
            text_position=config.LABEL_TEXT_POSITION,
            text_scale=config.LABEL_TEXT_SCALE,
            text_thickness=config.LABEL_TEXT_THICKNESS
        )

        # Annotate ellipses and labels for this team/role
        try:
            annotated_frame = temp_ellipse_annotator.annotate(
                scene=annotated_frame,
                detections=team_detections
            )
            annotated_frame = temp_label_annotator.annotate(
                scene=annotated_frame,
                detections=team_detections,
                labels=team_labels
            )
        except Exception as e:
            # Log detailed error including team ID and color
            hex_color = team_color.as_hex() if hasattr(team_color, 'as_hex') else str(team_color)
            logger.error(f"Error during annotation for team/role ID {current_team_id} "
                         f"(Color: {hex_color}): {e}", exc_info=True)
            # Continue annotating other teams if possible

    return annotated_frame
