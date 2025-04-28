import numpy as np
import supervision as sv
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque
import logging
import time
import math
import torch # For cache clearing
import json # For saving results

from app.core import config
from app.processing.detection import ObjectDetector
from app.processing.classification import TeamClassifier, classify_detections
from app.processing.tracking import ObjectTracker
from app.processing.ocr import perform_ocr_on_crop
from app.processing.annotation import annotate_frame, draw_ball_trail, draw_current_ball_marker
from app.processing.utils import get_video_info

# Configure logging
logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Orchestrates the entire video processing pipeline:
    Detection -> Classification -> Tracking -> OCR -> Annotation.
    Manages state like player data and ball positions across frames.
    """
    def __init__(self, source_video_path: str, output_dir: str = None):
        """
        Initializes the VideoProcessor.

        Args:
            source_video_path (str): Path to the input video file.
            output_dir (str, optional): Directory to save the processed video and JSON.
                                        Defaults to config.DEFAULT_OUTPUT_DIR.
        """
        self.source_path = Path(source_video_path)
        self.output_dir = Path(output_dir) if output_dir else config.DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

        if not self.source_path.is_file():
            raise FileNotFoundError(f"Source video not found: {self.source_path}")

        # Get video info early
        self.video_info = get_video_info(str(self.source_path))
        if not self.video_info or not self.video_info.fps or self.video_info.fps <= 0:
            logger.warning("Could not determine valid FPS. Using default 30.")
            self.fps = 30
        else:
            self.fps = self.video_info.fps

        # Initialize components
        self.detector = ObjectDetector()
        self.team_classifier = TeamClassifier(device=config.DEVICE)
        self.tracker = ObjectTracker()

        # --- State Management ---
        # Player data: {track_id: {"jersey_id": str|None, "jersey_confidence": float|None,
        #                          "last_seen": int, "team_id": int,
        #                          "mismatch_history": deque}}
        self.player_data = {}
        # Recently lost jerseys: {jersey_num_str: deque([{"tracker_id": int, "last_seen": int, "team_id": int}, ...])}
        self.recently_lost_jerseys = defaultdict(lambda: deque(maxlen=10)) # Store last 10 sightings per jersey#
        # Ball positions: deque([(x, y), ...], maxlen=...)
        self.ball_positions = deque(maxlen=int(self.fps * config.BALL_TRAIL_SECONDS))
        # Lost track memory in frames
        self.lost_track_memory_frames = int(self.fps * config.LOST_TRACK_MEMORY_SECONDS)
        logger.info(f"Lost track memory set to {self.lost_track_memory_frames} frames ({config.LOST_TRACK_MEMORY_SECONDS} seconds)")

        # --- Output paths ---
        self.output_video_path = self.output_dir / f"{self.source_path.stem}_processed.mp4"
        self.output_json_path = self.output_dir / f"{self.source_path.stem}_results.json"

        # --- Processing Results ---
        self.processing_summary = {} # To store summary data for JSON output


    def _train_classifier(self):
        """Trains the team classifier using the source video."""
        if not self.team_classifier.is_trained:
            logger.info("Training team classifier...")
            start_time = time.time()
            self.team_classifier.train(str(self.source_path), self.detector)
            elapsed = time.time() - start_time
            if self.team_classifier.is_trained:
                logger.info(f"Team classifier trained successfully in {elapsed:.2f} seconds.")
            else:
                logger.warning(f"Team classifier training failed or was skipped after {elapsed:.2f} seconds.")
        else:
            logger.info("Team classifier already trained.")


    def _update_ball_trail(self, ball_detections: sv.Detections, frame_idx: int):
        """Updates the ball position deque with filtering for outliers."""
        if len(ball_detections) == 0:
            # No ball detected, don't update the trail for this frame
            return

        # Assume single ball, take the first/highest confidence detection
        # (Add logic here if multiple balls are possible)
        best_ball_idx = np.argmax(ball_detections.confidence) if ball_detections.confidence is not None else 0
        x1, y1, x2, y2 = ball_detections.xyxy[best_ball_idx]
        current_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        is_valid_position = True # Assume valid initially
        if len(self.ball_positions) > 0:
            # Compare with the last known valid position
            prev_center = self.ball_positions[-1]
            # Check if prev_center is a valid tuple before calculating distance
            if isinstance(prev_center, tuple) and len(prev_center) == 2:
                distance = math.dist(current_center, prev_center)
                # Check if distance exceeds threshold
                if distance > config.MAX_BALL_DISTANCE_PER_FRAME:
                    is_valid_position = False
                    # logger.debug(f"[Frame {frame_idx}] Ball position outlier detected. Dist: {distance:.1f} > {config.MAX_BALL_DISTANCE_PER_FRAME}. Skipping update.")
            # else: Invalid format in deque? Treat current as valid start.

        # Append only if the position is considered valid
        if is_valid_position:
            self.ball_positions.append(current_center)
        # If not valid, self.ball_positions is simply not updated for this frame


    def _manage_player_ids(self, frame: np.ndarray, tracked_detections: sv.Detections, frame_idx: int) -> list[str]:
        """
        Performs OCR on tracked detections, assigns jersey numbers, manages track
        continuity using jersey numbers, and generates final labels.

        Updates self.player_data and self.recently_lost_jerseys.

        Args:
            frame: The current video frame.
            tracked_detections: Detections output from the tracker.
            frame_idx: The current frame index.

        Returns:
            A list of final labels (e.g., "T1 P5 #23") for each tracked detection.
        """
        final_labels = []
        current_frame_player_data = {} # Track data for players seen in *this* frame
        current_frame_tracker_ids = set(tracked_detections.tracker_id) if len(tracked_detections) > 0 else set()

        if len(tracked_detections) == 0:
            # Handle lost tracks even if no detections in current frame
            self._update_lost_tracks(current_frame_tracker_ids, frame_idx)
            return []

        # --- Process each tracked detection ---
        for i in range(len(tracked_detections)):
            track_id = tracked_detections.tracker_id[i]
            team_id = tracked_detections.class_id[i]
            bbox = tracked_detections.xyxy[i]

            # Crop player image for OCR
            player_crop = sv.crop_image(image=frame, xyxy=bbox)
            detected_jersey_num, ocr_confidence = None, None
            if player_crop is not None and player_crop.size > 0:
                 # Perform OCR only on player/GK detections if needed (optional optimization)
                 # if team_id in [config.TEAM_A_ID, config.TEAM_B_ID]:
                 detected_jersey_num, ocr_confidence = perform_ocr_on_crop(
                     player_crop, frame_idx=frame_idx, track_id=track_id
                 )

            assigned_jersey_id = None # The final jersey ID assigned to this track for this frame

            # --- Logic for Existing Tracks ---
            if track_id in self.player_data:
                p_data = self.player_data[track_id]
                p_data["last_seen"] = frame_idx
                p_data["team_id"] = team_id # Update team ID in case classification changed
                current_jersey_id = p_data.get("jersey_id")
                mismatch_history = p_data.setdefault("mismatch_history", deque(maxlen=config.MISMATCH_CONSISTENCY_FRAMES))

                if detected_jersey_num is not None:
                    # Case 1: No previous jersey or OCR matches current -> Update/Confirm
                    if current_jersey_id is None or detected_jersey_num == current_jersey_id:
                        p_data["jersey_id"] = detected_jersey_num
                        p_data["jersey_confidence"] = ocr_confidence
                        mismatch_history.clear()
                    # Case 2: OCR mismatches current jersey -> Check consistency
                    else:
                        mismatch_history.append(detected_jersey_num)
                        # If mismatch is consistent, overwrite the old jersey ID
                        if (len(mismatch_history) >= config.MISMATCH_CONSISTENCY_FRAMES and
                            all(num == detected_jersey_num for num in mismatch_history)):
                            logger.info(f"[Frame {frame_idx}] Track {track_id}: Jersey ID updated from '{current_jersey_id}' to '{detected_jersey_num}' due to consistent mismatch.")
                            p_data["jersey_id"] = detected_jersey_num
                            p_data["jersey_confidence"] = ocr_confidence
                            mismatch_history.clear()
                            # Note: The old jersey ID is now potentially "lost"
                            # We could add logic here to add the old ID to recently_lost_jerseys
                            # if current_jersey_id is not None:
                            #    self.recently_lost_jerseys[current_jersey_id].append(...)
                else:
                    # No jersey detected in this frame, clear mismatch history
                    mismatch_history.clear()

                assigned_jersey_id = p_data.get("jersey_id") # Use the potentially updated ID
                current_frame_player_data[track_id] = p_data # Add to current frame data

            # --- Logic for New Tracks (or tracks previously lost) ---
            else:
                found_match = False
                # Try to re-identify using OCR'd jersey number from recently lost tracks
                if detected_jersey_num is not None and detected_jersey_num in self.recently_lost_jerseys:
                    potential_matches = []
                    # Iterate through lost tracks with the same jersey number
                    for lost_track_info in reversed(self.recently_lost_jerseys[detected_jersey_num]):
                        time_diff = frame_idx - lost_track_info["last_seen"]
                        # Check if within time window and team ID matches (important!)
                        if time_diff < self.lost_track_memory_frames and lost_track_info["team_id"] == team_id:
                            potential_matches.append((lost_track_info, time_diff))

                    if potential_matches:
                        # Find the most recent match
                        potential_matches.sort(key=lambda x: x[1]) # Sort by time_diff (ascending)
                        best_match_info, _ = potential_matches[0]
                        original_track_id = best_match_info["tracker_id"]

                        logger.info(f"[Frame {frame_idx}] Re-identified Track {track_id} as previous Track {original_track_id} (Jersey: {detected_jersey_num}, Team: {team_id})")

                        # Re-instate the data from the lost track
                        assigned_jersey_id = detected_jersey_num
                        p_data = {
                            "jersey_id": assigned_jersey_id,
                            "jersey_confidence": ocr_confidence,
                            "last_seen": frame_idx,
                            "team_id": team_id,
                            "mismatch_history": deque(maxlen=config.MISMATCH_CONSISTENCY_FRAMES)
                            # Potentially copy other relevant data from best_match_info if needed
                        }
                        current_frame_player_data[track_id] = p_data
                        found_match = True

                        # Remove the matched entry from recently_lost_jerseys
                        try:
                            self.recently_lost_jerseys[detected_jersey_num].remove(best_match_info)
                        except ValueError:
                            logger.warning(f"Could not remove re-identified track {original_track_id} from recently_lost_jerseys[{detected_jersey_num}].")


                # If no match found via re-identification, treat as a new player entry
                if not found_match:
                    assigned_jersey_id = detected_jersey_num
                    current_frame_player_data[track_id] = {
                        "jersey_id": assigned_jersey_id,
                        "jersey_confidence": ocr_confidence if detected_jersey_num is not None else None,
                        "last_seen": frame_idx,
                        "team_id": team_id,
                        "mismatch_history": deque(maxlen=config.MISMATCH_CONSISTENCY_FRAMES)
                    }

            # --- Generate Label ---
            if team_id == config.TEAM_A_ID: team_prefix = "T1"
            elif team_id == config.TEAM_B_ID: team_prefix = "T2"
            elif team_id == config.REFEREE_TEAM_ID: team_prefix = "Ref"
            else: team_prefix = f"T?{team_id}" # Unknown team

            base_label = f"{team_prefix} P{track_id}"
            display_id = base_label
            if assigned_jersey_id is not None:
                display_id = f"{base_label} #{assigned_jersey_id}"
            final_labels.append(display_id)

        # --- Update Global State ---
        self.player_data = current_frame_player_data
        self._update_lost_tracks(current_frame_tracker_ids, frame_idx)

        return final_labels


    def _update_lost_tracks(self, current_frame_tracker_ids: set, frame_idx: int):
        """Identifies lost tracks and updates the recently_lost_jerseys registry."""
        lost_tracker_ids = set(self.player_data.keys()) - current_frame_tracker_ids
        for lost_id in lost_tracker_ids:
            lost_info = self.player_data.get(lost_id) # Use .get for safety
            if lost_info and lost_info.get("jersey_id") is not None:
                # Add info about the lost track to the deque for that jersey number
                jersey_id = lost_info["jersey_id"]
                self.recently_lost_jerseys[jersey_id].append({
                    "tracker_id": lost_id,
                    "last_seen": lost_info["last_seen"], # Use the last frame it was seen
                    "team_id": lost_info["team_id"]
                })
                # logger.debug(f"[Frame {frame_idx}] Track {lost_id} (Jersey: {jersey_id}) lost. Added to recently_lost_jerseys.")

        # --- Periodic Cleanup of recently_lost_jerseys ---
        # Clean up very old entries occasionally to prevent memory leaks
        # Check every minute of video (adjust frequency as needed)
        cleanup_interval_frames = int(self.fps * 60)
        if frame_idx > 0 and cleanup_interval_frames > 0 and frame_idx % cleanup_interval_frames == 0:
            logger.debug(f"[Frame {frame_idx}] Performing periodic cleanup of recently_lost_jerseys...")
            cutoff_frame = frame_idx - (self.lost_track_memory_frames * 2) # Keep entries for double the memory time
            keys_to_delete = []
            for jersey_num, track_deque in self.recently_lost_jerseys.items():
                # Create a new deque with only valid entries
                valid_entries = deque(
                    [entry for entry in track_deque if entry["last_seen"] >= cutoff_frame],
                    maxlen=track_deque.maxlen # Keep original maxlen
                )
                if valid_entries:
                    self.recently_lost_jerseys[jersey_num] = valid_entries
                else:
                    keys_to_delete.append(jersey_num) # Mark for deletion if deque becomes empty

            for key in keys_to_delete:
                del self.recently_lost_jerseys[key]
            logger.debug(f"Cleanup complete. Removed {len(keys_to_delete)} empty jersey entries.")


    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray | None:
        """Processes a single frame through the pipeline."""
        try:
            # 1. Detection
            detections = self.detector.predict(frame)
            if len(detections) == 0:
                # Still need to update tracker with empty and handle lost tracks
                 tracked_detections = self.tracker.update(frame, sv.Detections.empty())
                 final_labels = self._manage_player_ids(frame, tracked_detections, frame_idx)
                 # Annotate only ball trail if needed
                 annotated_frame = frame.copy()
                 if self.ball_positions:
                     annotated_frame = draw_ball_trail(annotated_frame, self.ball_positions)
                     annotated_frame = draw_current_ball_marker(annotated_frame, self.ball_positions)
                 return annotated_frame # Return frame with potential ball trail

            # 2. Ball Position Update (before classification/tracking filters ball)
            ball_detections = detections[detections.class_id == config.BALL_ID]
            self._update_ball_trail(ball_detections, frame_idx)

            # 3. Classification (Players, Goalkeepers, Referees)
            classified_detections, dynamic_color_map = classify_detections(
                frame, detections, self.team_classifier
            )

            # 4. Tracking
            tracked_detections = self.tracker.update(frame, classified_detections)

            # 5. OCR & Player ID Management (Label Generation)
            final_labels = self._manage_player_ids(frame, tracked_detections, frame_idx)

            # 6. Annotation
            annotated_frame = annotate_frame(
                frame,
                tracked_detections,
                final_labels,
                dynamic_color_map,
                self.ball_positions
            )

            return annotated_frame

        except Exception as e:
            logger.error(f"--- CRITICAL ERROR processing frame {frame_idx}: {e} ---", exc_info=True)
            # Return the original frame to avoid crashing the video writing process
            return frame


    def run(self):
        """Runs the full video processing pipeline."""
        logger.info(f"Starting video processing for: {self.source_path}")
        start_time = time.time()

        # --- Setup ---
        # Train classifier if not already done (e.g., if this instance is reused)
        self._train_classifier()

        # Video reader and writer
        frame_generator = sv.get_video_frames_generator(source_path=str(self.source_path), stride=1)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or use 'avc1' etc.
        video_writer = cv2.VideoWriter(
            str(self.output_video_path),
            fourcc,
            self.fps,
            (self.video_info.width, self.video_info.height)
        )

        if not video_writer.isOpened():
             logger.error(f"Failed to open video writer for path: {self.output_video_path}")
             return None, None # Indicate failure

        # --- Processing Loop ---
        total_frames = self.video_info.total_frames if self.video_info.total_frames else None
        logger.info(f"Processing {total_frames or 'unknown'} frames...")
        frame_count = 0
        processing_errors = 0

        try:
            with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
                for frame_idx, frame in enumerate(frame_generator):
                    if frame is None:
                        logger.warning(f"Received None frame at index {frame_idx}, ending processing.")
                        break

                    annotated_frame = self._process_frame(frame, frame_idx)

                    if annotated_frame is not None:
                        video_writer.write(annotated_frame)
                    else:
                        # This case should be handled within _process_frame by returning original
                        logger.error(f"process_frame returned None unexpectedly for frame {frame_idx}. Writing original frame.")
                        video_writer.write(frame)
                        processing_errors += 1

                    pbar.update(1)
                    frame_count = frame_idx + 1 # Keep track of processed frames

        except KeyboardInterrupt:
            logger.warning("\nProcessing interrupted by user.")
        except Exception as e:
            logger.error(f"\n--- UNHANDLED EXCEPTION in main processing loop: {e} ---", exc_info=True)
            processing_errors += 1 # Count as an error
        finally:
            video_writer.release()
            logger.info(f"Video writer released for {self.output_video_path}")
            # Clear CUDA cache after processing
            if config.DEVICE.type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared.")
                except Exception as e:
                    logger.warning(f"Error clearing CUDA cache: {e}")

        # --- Finalization ---
        end_time = time.time()
        processing_time = end_time - start_time
        avg_fps = frame_count / processing_time if processing_time > 0 else 0
        logger.info(f"\nFinished processing {frame_count} frames in {processing_time:.2f} seconds (Avg FPS: {avg_fps:.2f}).")
        logger.info(f"Annotated video saved to: {self.output_video_path}")
        if processing_errors > 0:
             logger.warning(f"{processing_errors} errors occurred during frame processing.")

        # --- Generate Summary JSON ---
        self.processing_summary = {
            "source_video": str(self.source_path.name),
            "processed_video": str(self.output_video_path.name),
            "status": "Completed" if processing_errors == 0 else f"Completed with {processing_errors} errors",
            "total_frames_processed": frame_count,
            "processing_time_seconds": round(processing_time, 2),
            "average_fps": round(avg_fps, 2),
            "output_directory": str(self.output_dir),
            "final_player_data": self._prepare_player_data_for_json() # Get final state
        }
        self._save_summary()

        return str(self.output_video_path), str(self.output_json_path)

    def _prepare_player_data_for_json(self):
        """Converts internal player data (with deques) to JSON-serializable format."""
        serializable_data = {}
        for track_id, data in self.player_data.items():
            serializable_data[str(track_id)] = { # Use string keys for JSON
                "jersey_id": data.get("jersey_id"),
                "jersey_confidence": data.get("jersey_confidence"),
                "last_seen_frame": data.get("last_seen"),
                "team_id": data.get("team_id"),
                # Convert mismatch deque to list for JSON
                "mismatch_history": list(data.get("mismatch_history", []))
            }
        return serializable_data

    def _save_summary(self):
        """Saves the processing summary to a JSON file."""
        try:
            with open(self.output_json_path, 'w') as f:
                json.dump(self.processing_summary, f, indent=4)
            logger.info(f"Processing summary saved to: {self.output_json_path}")
        except Exception as e:
            logger.error(f"Failed to save processing summary JSON: {e}", exc_info=True)

