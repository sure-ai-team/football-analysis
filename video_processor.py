# video_processor.py
"""
Contains the main frame processing logic, including detection, tracking,
classification, OCR, and annotation.
"""
import supervision as sv
import numpy as np
from boxmot import BotSort # Using BoTSORT
import cv2
from collections import defaultdict, deque
import logging
import traceback
import random
import math
import torch

# Import from local modules
import config
import utils
# TeamClassifier is used for prediction here, assuming it's fitted beforehand
from team_classifier_setup import TeamClassifier # Or adjust import as needed

# --- Global State (managed within the processor context) ---
# These will be initialized/updated by the main script or a class wrapper if preferred
player_data = {} # Stores persistent data per track_id {track_id: {jersey_id, last_seen, team_id, ...}}
recently_lost_jerseys = defaultdict(lambda: deque(maxlen=10)) # {jersey_id: deque([{track_id, last_seen, team_id}, ...])}
ball_positions = None # Deque for storing recent ball positions (center coordinates)

# --- Tracker Initialization ---
def initialize_tracker():
    """Initializes the BoTSORT tracker."""
    logging.info("Initializing BoTSORT tracker...")
    logging.info(f"ReID Weights Path: {config.REID_WEIGHTS_PATH}")
    logging.info(f"Using ReID: {config.TRACKER_WITH_REID}")
    logging.info(f"Tracker Device: {config.DEVICE}")
    logging.info(f"Tracker Half Precision: {config.TRACKER_HALF_PRECISION}")
    try:
        tracker = BotSort(
            reid_weights=config.REID_WEIGHTS_PATH if config.TRACKER_WITH_REID else None,
            device=config.DEVICE,
            half=config.TRACKER_HALF_PRECISION,
            with_reid=config.TRACKER_WITH_REID,
        )
        logging.info("BoTSORT tracker initialized successfully.")
        return tracker
    except Exception as e:
        logging.error(f"Error initializing BoTSORT tracker: {e}", exc_info=True)
        return None

# --- Ball Trail Initialization ---
def initialize_ball_trail(fps: float):
    """Initializes the deque for ball trail based on FPS."""
    global ball_positions
    if not config.BALL_TRAIL_ENABLED:
        logging.warning("Ball trail is disabled in config.")
        ball_positions = None
        return

    if fps and fps > 0:
        trail_maxlen = int(fps * config.BALL_TRAIL_SECONDS)
        ball_positions = deque(maxlen=trail_maxlen)
        logging.info(f"Ball trail deque initialized with maxlen={trail_maxlen} ({config.BALL_TRAIL_SECONDS} seconds)")
    else:
        logging.warning("Could not determine valid FPS. Ball trail disabled.")
        ball_positions = None # Ensure it's None if FPS is invalid

# --- Frame Processing Function ---
def process_frame(
    frame: np.ndarray,
    frame_idx: int,
    player_detection_model, # YOLO model
    team_classifier: TeamClassifier | None, # Fitted TeamClassifier instance
    tracker: BotSort, # Initialized BoTSORT instance
    ocr_model, # Initialized OCR model (or None)
    ocr_available: bool, # Flag if OCR is usable
    video_info: sv.VideoInfo, # Video properties
    fps: float # Video FPS for time-based logic
):
    """
    Processes a single video frame for detection, tracking, classification, and annotation.

    Args:
        frame: The input video frame (NumPy array BGR).
        frame_idx: The index of the current frame.
        player_detection_model: The loaded YOLO detection model.
        team_classifier: The fitted TeamClassifier instance (can be None).
        tracker: The initialized BoTSORT tracker.
        ocr_model: The loaded OCR model (can be None).
        ocr_available: Boolean indicating if OCR can be used.
        video_info: Information about the video (width, height, etc.).
        fps: Frames per second of the video.

    Returns:
        The annotated frame (NumPy array BGR), or the original frame if errors occur.
    """
    global player_data, recently_lost_jerseys, ball_positions # Access global state

    if frame is None:
        logging.error(f"Received None frame at index {frame_idx}. Skipping processing.")
        return None # Or return a black frame of correct size?

    height, width = video_info.height, video_info.width
    LOST_TRACK_MEMORY_FRAMES = int(fps * config.LOST_TRACK_MEMORY_SECONDS) if fps > 0 else 30 * config.LOST_TRACK_MEMORY_SECONDS

    try:
        # 1. Detection
        results = player_detection_model.predict(
            frame,
            conf=config.DETECTION_CONFIDENCE_THRESHOLD,
            iou=config.DETECTION_NMS_THRESHOLD, # Using NMS threshold here as IoU for prediction
            device=config.DEVICE,
            verbose=False
        )
        if not results or len(results) == 0:
            logging.debug(f"No detections in frame {frame_idx}")
             # Still need to update tracker with empty input if tracker exists
            if tracker is not None:
                try: tracker.update(np.empty((0, 6)), frame)
                except Exception as e_trk: logging.warning(f"[Frame {frame_idx}] Error updating tracker with empty detections: {e_trk}")
            return frame # Return original frame if no detections

        detections = sv.Detections.from_ultralytics(results[0])
        logging.debug(f"[Frame {frame_idx}] Initial detections: {len(detections)}")

        # 2. Pre-processing & Ball Position Update
        ball_detections = detections[detections.class_id == config.BALL_ID]
        people_detections = detections[detections.class_id != config.BALL_ID]

        # --- Update ball trail deque with outlier filtering ---
        if config.BALL_TRAIL_ENABLED and ball_positions is not None and len(ball_detections) > 0:
            # Assuming single ball, take the first detection with highest confidence?
            # Or just the first one found
            ball_det_to_use = ball_detections[0:1] # Take the first one
            x1, y1, x2, y2 = ball_det_to_use.xyxy[0]
            current_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            is_valid_position = True # Assume valid initially
            if len(ball_positions) > 0:
                # Compare with the last known valid position in the deque
                prev_center = ball_positions[-1]
                if isinstance(prev_center, tuple) and len(prev_center) == 2:
                    distance = math.dist(current_center, prev_center)
                    if distance > config.MAX_BALL_DISTANCE_PER_FRAME:
                        is_valid_position = False
                        logging.debug(f"[Frame {frame_idx}] Ball position outlier detected. Dist: {distance:.1f} > {config.MAX_BALL_DISTANCE_PER_FRAME}. Skipping update.")
                # else: Invalid format in deque? Treat current as valid start.

            if is_valid_position:
                ball_positions.append(current_center)
            # If not valid, ball_positions is simply not updated for this frame

        # 3. Team/Role Classification
        players_detections = people_detections[people_detections.class_id == config.PLAYER_ID]
        goalkeepers_detections = people_detections[people_detections.class_id == config.GOALKEEPER_ID]
        referees_detections = people_detections[people_detections.class_id == config.REFEREE_ID]

        # --- Player Classification ---
        classified_players = sv.Detections.empty()
        if len(players_detections) > 0 and team_classifier is not None:
            players_crops = []; valid_indices = []
            for i, xyxy in enumerate(players_detections.xyxy):
                crop = sv.crop_image(frame, xyxy)
                if crop is not None and crop.size > 0:
                    players_crops.append(crop)
                    valid_indices.append(i)

            if players_crops:
                try:
                    # Predict team IDs using the fitted classifier
                    predicted_team_ids = team_classifier.predict(players_crops) # Returns list/array of team IDs
                    if predicted_team_ids is not None and len(predicted_team_ids) == len(players_crops):
                        # Create an array to hold assigned IDs, default to -1 (unclassified)
                        assigned_ids = np.full(len(players_detections), -1, dtype=int)
                        # Map predicted IDs back to the original detection indices
                        for i, pred_id in enumerate(predicted_team_ids):
                            assigned_ids[valid_indices[i]] = pred_id # Assuming pred_id is TEAM_A_ID or TEAM_B_ID

                        # Update class_id in the Detections object
                        players_detections.class_id = assigned_ids
                        # Filter out players that couldn't be classified (id still -1)
                        valid_classification_mask = (players_detections.class_id != -1)
                        classified_players = players_detections[valid_classification_mask]
                        logging.debug(f"[Frame {frame_idx}] Classified {len(classified_players)} players.")
                    else:
                        logging.warning(f"[Frame {frame_idx}] Team classifier prediction mismatch or returned None.")
                        # Players remain unclassified (will be tracked with original PLAYER_ID if not filtered)
                except Exception as e_cls:
                    logging.error(f"[Frame {frame_idx}] Error during team classification prediction: {e_cls}", exc_info=True)
        elif team_classifier is None:
            logging.debug(f"[Frame {frame_idx}] Team classifier not available, skipping player classification.")
            # Keep original player detections if no classifier
            classified_players = players_detections # Treat them as classified with PLAYER_ID


        # --- Calculate Dynamic Team Colors (based on classified players) ---
        team_a_detections = classified_players[classified_players.class_id == config.TEAM_A_ID]
        team_b_detections = classified_players[classified_players.class_id == config.TEAM_B_ID]

        current_team_a_color = utils.calculate_average_color(frame, team_a_detections) or config.DEFAULT_TEAM_A_COLOR
        current_team_b_color = utils.calculate_average_color(frame, team_b_detections) or config.DEFAULT_TEAM_B_COLOR
        current_referee_color = config.DEFAULT_REFEREE_COLOR # Referee color is fixed for now

        # Map team IDs to their current calculated/default colors
        dynamic_color_map = {
            config.TEAM_A_ID: current_team_a_color,
            config.TEAM_B_ID: current_team_b_color,
            config.REFEREE_TEAM_ID: current_referee_color,
            config.PLAYER_ID: config.FALLBACK_COLOR, # Color for unclassified players
            config.GOALKEEPER_ID: config.FALLBACK_COLOR # Fallback for GKs if resolution fails
        }

        # --- Goalkeeper Classification (using enhanced function) ---
        classified_gks = sv.Detections.empty()
        if len(goalkeepers_detections) > 0:
            gk_team_ids = utils.resolve_goalkeepers_team_id(
                frame,
                goalkeepers_detections,
                current_team_a_color, # Pass calculated color
                current_team_b_color  # Pass calculated color
            )
            if gk_team_ids is not None and len(gk_team_ids) == len(goalkeepers_detections):
                valid_gk_mask = (gk_team_ids != -1) # Filter out GKs where resolution failed (-1)
                goalkeepers_detections.class_id = gk_team_ids # Assign resolved team IDs
                classified_gks = goalkeepers_detections[valid_gk_mask] # Only keep successfully classified GKs
                logging.debug(f"[Frame {frame_idx}] Resolved {len(classified_gks)} goalkeepers.")
            else:
                logging.warning(f"[Frame {frame_idx}] Goalkeeper resolution returned unexpected result or failed for all.")


        # --- Referee Classification (Assign fixed Referee Team ID) ---
        classified_refs = sv.Detections.empty()
        if len(referees_detections) > 0:
            ref_team_ids = np.full(len(referees_detections), config.REFEREE_TEAM_ID)
            referees_detections.class_id = ref_team_ids
            classified_refs = referees_detections
            logging.debug(f"[Frame {frame_idx}] Identified {len(classified_refs)} referees.")

        # --- Merge All Classified/Identified People Detections for Tracking ---
        # Includes classified players, resolved GKs, and identified referees
        detections_to_track = sv.Detections.merge([classified_players, classified_gks, classified_refs])
        logging.debug(f"[Frame {frame_idx}] Detections merged for tracking: {len(detections_to_track)}")


        # 4. Tracking using BoTSORT
        tracked_detections = sv.Detections.empty()
        current_frame_tracker_ids = set()
        if len(detections_to_track) > 0 and tracker is not None:
            # BoTSORT expects input in format: [x1, y1, x2, y2, conf, cls_id]
            boxmot_input = np.hstack((
                detections_to_track.xyxy,
                detections_to_track.confidence[:, np.newaxis],
                detections_to_track.class_id[:, np.newaxis] # Use the *assigned* team/role IDs
            ))
            try:
                # Update the tracker
                tracks = tracker.update(boxmot_input, frame) # Pass the original frame image

                if tracks.shape[0] > 0:
                    # BoTSORT output format: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
                    tracked_detections = sv.Detections(
                        xyxy=tracks[:, 0:4],
                        tracker_id=tracks[:, 4].astype(int),
                        confidence=tracks[:, 5],
                        class_id=tracks[:, 6].astype(int) # Get the class ID assigned by the tracker
                    )
                    current_frame_tracker_ids = set(tracked_detections.tracker_id)
                    logging.debug(f"[Frame {frame_idx}] Tracked detections: {len(tracked_detections)}")
                else:
                     logging.debug(f"[Frame {frame_idx}] Tracker updated but returned no tracks.")

            except Exception as e_trk:
                logging.error(f"[Frame {frame_idx}] Error during tracker update: {e_trk}", exc_info=True)
                tracked_detections = sv.Detections.empty() # Ensure empty if error

        elif tracker is not None:
            # Update tracker with empty input if no detections to track
            try:
                tracker.update(np.empty((0, 6)), frame)
                logging.debug(f"[Frame {frame_idx}] Updated tracker with empty input.")
            except Exception as e_trk:
                logging.warning(f"[Frame {frame_idx}] Error updating tracker with empty input: {e_trk}")

        # 5. OCR and Player ID Management (Label Generation)
        final_labels = []
        current_player_data = {} # Track data for players seen in *this* frame
        if len(tracked_detections) > 0:
            for i in range(len(tracked_detections)):
                track_id = tracked_detections.tracker_id[i]
                team_id = tracked_detections.class_id[i] # Team/Role ID from tracker
                bbox = tracked_detections.xyxy[i]

                detected_jersey_num: str | None = None
                ocr_confidence: float | None = None
                assigned_jersey_id: str | None = None # Final jersey # for this track in this frame

                # --- Perform OCR only if enabled and not a referee ---
                if config.OCR_ENABLED and ocr_available and team_id != config.REFEREE_TEAM_ID:
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1) # Clamp coordinates
                    x2, y2 = min(width, x2), min(height, y2)

                    if x1 < x2 and y1 < y2: # Check for valid bbox dimensions
                        player_crop = frame[y1:y2, x1:x2]
                        # Convert to grayscale for OCR (optional, depends on OCR model needs)
                        # gray_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
                        # Perform OCR on the BGR crop (as per utils function)
                        detected_jersey_num, ocr_confidence = utils.perform_ocr_on_crop(ocr_model, player_crop, ocr_available)

                        # Save debug crops if number detected
                        if detected_jersey_num is not None and config.OCR_DEBUG_DIR:
                             # Pass BGR crop and the input to OCR (which is also BGR now)
                             utils.save_ocr_debug_crop(frame_idx, track_id, player_crop, player_crop)
                    else:
                         logging.warning(f"[Frame {frame_idx}] Invalid bbox dimensions for track {track_id}, skipping crop/OCR.")


                # --- Manage Jersey ID Persistence and Re-identification ---
                if track_id in player_data:
                    # Player track already exists
                    p_data = player_data[track_id]
                    p_data["last_seen"] = frame_idx
                    p_data["team_id"] = team_id # Update team ID in case it changed (e.g., GK resolution)
                    current_jersey_id = p_data.get("jersey_id") # Use .get for safety
                    mismatch_history = p_data.get("mismatch_history", deque(maxlen=config.MISMATCH_CONSISTENCY_FRAMES)) # Get or create

                    if detected_jersey_num is not None:
                        # OCR detected a number for this known track
                        if current_jersey_id is None or detected_jersey_num == current_jersey_id:
                            # First time seeing number, or it matches the stored one
                            p_data["jersey_id"] = detected_jersey_num
                            p_data["jersey_confidence"] = ocr_confidence
                            mismatch_history.clear() # Reset mismatch counter
                        else:
                            # Mismatch detected!
                            mismatch_history.append(detected_jersey_num)
                            # Check if mismatch is consistent
                            if (len(mismatch_history) >= config.MISMATCH_CONSISTENCY_FRAMES and
                                all(num == detected_jersey_num for num in mismatch_history)):
                                logging.info(f"[Frame {frame_idx}] Track {track_id}: Jersey ID updated from {current_jersey_id} to {detected_jersey_num} after consistent mismatch.")
                                p_data["jersey_id"] = detected_jersey_num
                                p_data["jersey_confidence"] = ocr_confidence
                                mismatch_history.clear() # Reset after successful update
                            # else: Keep old jersey ID for now
                    else:
                        # No number detected by OCR in this frame for this known track
                        mismatch_history.clear() # Reset mismatch counter if OCR fails

                    p_data["mismatch_history"] = mismatch_history # Store updated history
                    assigned_jersey_id = p_data.get("jersey_id") # Use the potentially updated jersey ID
                    current_player_data[track_id] = p_data # Add updated data to this frame's dict

                else:
                    # New track ID encountered
                    found_match = False
                    # --- Attempt Re-identification using recently lost jerseys ---
                    if detected_jersey_num is not None and detected_jersey_num in recently_lost_jerseys:
                        potential_matches = []
                        # Iterate through tracks recently lost *with the same detected jersey number*
                        lost_queue = recently_lost_jerseys[detected_jersey_num]
                        for lost_track_info in reversed(list(lost_queue)): # Iterate copy in reverse
                            time_diff = frame_idx - lost_track_info["last_seen"]
                            # Check time window and if team ID matches (important!)
                            if time_diff < LOST_TRACK_MEMORY_FRAMES and lost_track_info.get("team_id") == team_id:
                                potential_matches.append((lost_track_info, time_diff))

                        if potential_matches:
                            # Found potential matches, sort by time difference (closest first)
                            potential_matches.sort(key=lambda x: x[1])
                            best_match_info, time_diff_match = potential_matches[0]
                            original_track_id = best_match_info["tracker_id"]

                            logging.info(f"[Frame {frame_idx}] Re-identified Track {track_id} as previous Track {original_track_id} (Jersey: {detected_jersey_num}, Team: {team_id}) after {time_diff_match} frames.")

                            assigned_jersey_id = detected_jersey_num # Assign the matched jersey number
                            # Create new data entry for the *new* track_id, inheriting the jersey number
                            p_data = {
                                "jersey_id": assigned_jersey_id,
                                "jersey_confidence": ocr_confidence,
                                "last_seen": frame_idx,
                                "team_id": team_id,
                                "mismatch_history": deque(maxlen=config.MISMATCH_CONSISTENCY_FRAMES)
                                # Optionally copy other relevant info from best_match_info if needed
                            }
                            current_player_data[track_id] = p_data
                            # Remove the matched entry from the lost queue to prevent re-matching
                            try:
                                lost_queue.remove(best_match_info)
                            except ValueError:
                                pass # Should not happen if iterating correctly, but safe check
                            found_match = True


                    # --- If no re-identification match, create new player entry ---
                    if not found_match:
                        assigned_jersey_id = detected_jersey_num # Assign if detected, else None
                        current_player_data[track_id] = {
                            "jersey_id": assigned_jersey_id,
                            "jersey_confidence": ocr_confidence if detected_jersey_num is not None else None,
                            "last_seen": frame_idx,
                            "team_id": team_id,
                            "mismatch_history": deque(maxlen=config.MISMATCH_CONSISTENCY_FRAMES)
                        }
                        if assigned_jersey_id:
                             logging.debug(f"[Frame {frame_idx}] New track {track_id} detected with jersey {assigned_jersey_id} (Team: {team_id}).")
                        else:
                             logging.debug(f"[Frame {frame_idx}] New track {track_id} detected without jersey (Team: {team_id}).")


                # --- Generate Label Text ---
                # Determine team prefix based on assigned team/role ID
                if team_id == config.TEAM_A_ID: team_prefix = "T1"
                elif team_id == config.TEAM_B_ID: team_prefix = "T2"
                elif team_id == config.REFEREE_TEAM_ID: team_prefix = "Ref"
                elif team_id == config.PLAYER_ID: team_prefix = "Player" # Unclassified player
                elif team_id == config.GOALKEEPER_ID: team_prefix = "GK" # Unresolved GK
                else: team_prefix = f"T{team_id}" # Fallback for unexpected IDs

                # Construct base label with track ID
                base_label = f"{team_prefix} {track_id}"
                display_id = base_label
                # Append jersey number if available
                if assigned_jersey_id is not None:
                    display_id = f"{base_label} #{assigned_jersey_id}"

                final_labels.append(display_id)

        # 6. Update Global Player Data & Handle Lost Tracks
        # Identify tracks that were present in the previous frame but not in the current one
        lost_tracker_ids = set(player_data.keys()) - current_frame_tracker_ids
        for lost_id in lost_tracker_ids:
            lost_info = player_data.get(lost_id) # Use .get for safety
            if lost_info:
                lost_jersey = lost_info.get("jersey_id")
                # If the lost track had an associated jersey number, add it to the recently lost queue
                if lost_jersey is not None:
                    # Store relevant info for potential re-identification
                    recently_lost_jerseys[lost_jersey].append({
                        "tracker_id": lost_id, # The ID that was just lost
                        "last_seen": lost_info.get("last_seen", frame_idx -1), # Frame it was last seen
                        "team_id": lost_info.get("team_id") # Team ID when last seen
                    })
                    logging.debug(f"[Frame {frame_idx}] Track {lost_id} (Jersey: {lost_jersey}) lost. Added to potential re-ID queue.")
                else:
                     logging.debug(f"[Frame {frame_idx}] Track {lost_id} (No jersey) lost.")


        # --- Periodic Cleanup of Old Entries in recently_lost_jerseys ---
        # Run cleanup less frequently (e.g., every minute) to avoid overhead
        if fps > 0 and frame_idx > 0 and frame_idx % (int(fps) * 60) == 0:
            logging.info(f"[Frame {frame_idx}] Performing periodic cleanup of lost jersey queue...")
            cleaned_count = 0
            current_time_limit = frame_idx - LOST_TRACK_MEMORY_FRAMES * 2 # Use a slightly larger buffer for cleanup
            for jersey_num in list(recently_lost_jerseys.keys()): # Iterate over keys copy
                q = recently_lost_jerseys[jersey_num]
                # Create a new deque with only entries within the time limit
                valid_entries = deque(
                    [entry for entry in q if entry.get("last_seen", 0) > current_time_limit],
                    maxlen=10 # Keep maxlen constraint
                )
                if valid_entries:
                    recently_lost_jerseys[jersey_num] = valid_entries
                else:
                    # If no valid entries remain, remove the jersey number key
                    del recently_lost_jerseys[jersey_num]
                    cleaned_count += 1
            logging.info(f"Cleanup complete. Removed {cleaned_count} expired jersey queues.")


        # Update the main player_data dictionary with data from the current frame
        player_data = current_player_data

        # 7. Annotation
        annotated_frame = frame.copy() # Work on a copy

        # --- Annotate "Magical" Ball Trail ---
        if config.BALL_TRAIL_ENABLED and ball_positions is not None and len(ball_positions) >= 2:
            num_points = len(ball_positions)
            for i in range(1, num_points):
                pt1 = ball_positions[i-1]
                pt2 = ball_positions[i]
                # Ensure points are valid tuples before drawing
                if isinstance(pt1, tuple) and isinstance(pt2, tuple) and len(pt1) == 2 and len(pt2) == 2:
                    # Draw trail line
                    cv2.line(annotated_frame, pt1, pt2, config.BALL_TRAIL_BASE_COLOR, config.BALL_TRAIL_THICKNESS, lineType=cv2.LINE_AA) # Smoother line

                    # --- Add Sparkles ---
                    # Calculate alpha for intensity fade (optional)
                    alpha_fraction = (i - 1) / max(1, num_points - 1) # Normalize index
                    sparkle_intensity = int(config.SPARKLE_BASE_INTENSITY + (config.SPARKLE_MAX_INTENSITY - config.SPARKLE_BASE_INTENSITY) * alpha_fraction)
                    sparkle_color = (sparkle_intensity, sparkle_intensity, sparkle_intensity) # Grayscale sparkle

                    # Draw multiple sparkles around the point pt2
                    for _ in range(config.SPARKLE_COUNT):
                        offset_x = random.randint(-config.SPARKLE_OFFSET, config.SPARKLE_OFFSET)
                        offset_y = random.randint(-config.SPARKLE_OFFSET, config.SPARKLE_OFFSET)
                        sparkle_pt = (pt2[0] + offset_x, pt2[1] + offset_y)
                        # Clamp sparkle points to frame boundaries
                        sparkle_pt_clamped = (
                            max(0, min(width - 1, sparkle_pt[0])),
                            max(0, min(height - 1, sparkle_pt[1]))
                        )
                        cv2.circle(annotated_frame, sparkle_pt_clamped, config.SPARKLE_RADIUS, sparkle_color, -1) # Filled circle

        # --- Annotate Current Ball Position ---
        if config.BALL_TRAIL_ENABLED and ball_positions is not None and len(ball_positions) > 0:
            last_pos = ball_positions[-1]
            if isinstance(last_pos, tuple) and len(last_pos) == 2:
                # Clamp position just in case
                last_pos_clamped = (
                     max(0, min(width - 1, last_pos[0])),
                     max(0, min(height - 1, last_pos[1]))
                )
                cv2.circle(
                    annotated_frame,
                    last_pos_clamped,
                    config.CURRENT_BALL_MARKER_RADIUS,
                    config.CURRENT_BALL_MARKER_COLOR,
                    config.CURRENT_BALL_MARKER_THICKNESS # Filled
                )


        # --- Annotate Tracked People (Players, GKs, Refs) ---
        if len(tracked_detections) > 0:
            # Ensure labels match detections count before proceeding
            if len(final_labels) != len(tracked_detections):
                 logging.error(f"[Frame {frame_idx}] Mismatch between number of tracked detections ({len(tracked_detections)}) and generated labels ({len(final_labels)}). Skipping annotation for this frame.")
            else:
                # Annotate detections team by team using their dynamic/default colors
                unique_team_ids = np.unique(tracked_detections.class_id)

                for current_team_id in unique_team_ids:
                    team_mask = (tracked_detections.class_id == current_team_id)
                    team_detections = tracked_detections[team_mask]
                    # Get labels corresponding to this team's detections
                    team_labels = [label for i, label in enumerate(final_labels) if team_mask[i]]

                    if len(team_detections) == 0: continue # Skip if no detections for this team ID

                    # Get the appropriate color from the dynamic map or fallback
                    team_color = dynamic_color_map.get(current_team_id, config.FALLBACK_COLOR)

                    # Create annotators with the specific team color
                    # Use try-except for individual annotator steps for robustness
                    try:
                        ellipse_annotator = sv.EllipseAnnotator(
                            color=team_color,
                            thickness=config.ELLIPSE_THICKNESS,
                            color_lookup=sv.ColorLookup.INDEX # Ensure color is used directly
                        )
                        annotated_frame = ellipse_annotator.annotate(annotated_frame, team_detections)
                    except Exception as e_ellipse:
                         logging.error(f"[Frame {frame_idx}] Error during ellipse annotation for team {current_team_id}: {e_ellipse}")

                    try:
                        label_annotator = sv.LabelAnnotator(
                            color=team_color, # Background color of label
                            text_color=config.LABEL_TEXT_COLOR, # Color of text
                            text_position=config.LABEL_TEXT_POSITION,
                            text_scale=config.LABEL_TEXT_SCALE,
                            text_thickness=config.LABEL_TEXT_THICKNESS,
                            color_lookup=sv.ColorLookup.INDEX # Ensure color is used directly
                        )
                        annotated_frame = label_annotator.annotate(annotated_frame, team_detections, labels=team_labels)
                    except Exception as e_label:
                         logging.error(f"[Frame {frame_idx}] Error during label annotation for team {current_team_id}: {e_label}")

        return annotated_frame

    except Exception as e:
        logging.error(f"--- CRITICAL ERROR processing frame {frame_idx}: {e} ---")
        logging.error(traceback.format_exc()) # Log the full traceback
        # Return the original frame in case of a major error to avoid crashing the video writer
        return frame

