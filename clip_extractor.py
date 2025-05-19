# clip_extractor.py
import cv2
import os
import supervision as sv
from collections import deque
import numpy as np
import logging
import config # Assuming config.py is updated

class ClipExtractor:
    def __init__(self, fps: float, video_info: sv.VideoInfo, output_dir: str):
        self.fps = fps
        self.video_info = video_info
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"ClipExtractor initialized. Output directory: {self.output_dir}, FPS: {self.fps}")

        # Load timing and threshold parameters from config
        self.seconds_buffer_pre_interaction = config.CLIP_SECONDS_BEFORE_INTERACTION
        self.seconds_buffer_post_interaction = config.CLIP_SECONDS_AFTER_INTERACTION
        self.min_total_clip_length_seconds = 4.0  # As per requirement "not less than 4 seconds"
        self.proximity_threshold_pixels = config.PROXIMITY_THRESHOLD_PIXELS
        self.interaction_iou_threshold = config.INTERACTION_IOU_THRESHOLD

        self.frames_pre_interaction_buffer = int(self.seconds_buffer_pre_interaction * self.fps)
        self.frames_post_interaction_buffer = int(self.seconds_buffer_post_interaction * self.fps)
        self.min_total_clip_frames = int(self.min_total_clip_length_seconds * self.fps)

        # Minimum duration an actual interaction must last to be considered for clipping
        self.min_interaction_duration_frames = int(config.MIN_INTERACTION_DURATION_SECONDS * self.fps)
        if self.min_interaction_duration_frames <= 0:
            self.min_interaction_duration_frames = 1
            logging.warning(f"MIN_INTERACTION_DURATION_SECONDS resulted in <=0 frames, defaulting to 1 frame.")
        
        # Frame buffer size calculation
        # Max expected duration of the core interaction itself (e.g., player possessing ball)
        # This helps determine buffer size, not clip length directly.
        max_expected_interaction_core_seconds = 5.0 # Configurable: Longest typical interaction
        max_expected_interaction_core_frames = int(max_expected_interaction_core_seconds * self.fps)

        # The buffer needs to hold enough frames for the longest possible clip definition
        # This considers: pre_buffer + core_interaction_max + post_buffer OR min_total_clip_frames
        max_natural_clip_frames = self.frames_pre_interaction_buffer + \
                                  max_expected_interaction_core_frames + \
                                  self.frames_post_interaction_buffer
        
        required_frames_for_longest_clip = max(max_natural_clip_frames, self.min_total_clip_frames)
        
        # Add a small safety margin to the buffer (e.g., 1 second of frames)
        self.frame_buffer_size = required_frames_for_longest_clip + int(1.0 * self.fps)
        
        self.annotated_frame_buffer = deque(maxlen=self.frame_buffer_size) # Stores {'frame_idx': frame_idx, 'frame_data': annotated_frame}

        self.active_interactions = {} # {player_track_id: {'start_frame': frame_idx, 'last_active_frame': frame_idx, 'ball_bbox_at_start': None}}
        self.clip_event_counter = 0
        
        logging.info(f"Interaction IOU threshold: {self.interaction_iou_threshold}")
        logging.info(f"Interaction proximity threshold: {self.proximity_threshold_pixels} pixels")
        logging.info(f"Min interaction duration to qualify: {self.min_interaction_duration_frames} frames ({config.MIN_INTERACTION_DURATION_SECONDS}s)")
        logging.info(f"Clip pre-interaction buffer: {self.frames_pre_interaction_buffer} frames ({self.seconds_buffer_pre_interaction}s)")
        logging.info(f"Clip post-interaction buffer: {self.frames_post_interaction_buffer} frames ({self.seconds_buffer_post_interaction}s)")
        logging.info(f"Minimum total clip length: {self.min_total_clip_frames} frames ({self.min_total_clip_length_seconds}s)")
        logging.info(f"Frame buffer size: {self.frame_buffer_size} frames")

    def _calculate_iou(self, box1, box2):
        # box: [x1, y1, x2, y2]
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height
        
        if inter_area == 0:
            return 0.0

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
            
        return inter_area / union_area
        
    def _calculate_proximity(self, box1, box2):
        """
        Calculate the minimum distance between the edges of two bounding boxes.
        Returns 0 if they overlap, otherwise the positive distance in pixels.
        """
        # Calculate centers of each box
        box1_center_x = (box1[0] + box1[2]) / 2
        box1_center_y = (box1[1] + box1[3]) / 2
        box2_center_x = (box2[0] + box2[2]) / 2
        box2_center_y = (box2[1] + box2[3]) / 2
        
        # Calculate Euclidean distance between centers
        center_distance = np.sqrt((box1_center_x - box2_center_x)**2 + (box1_center_y - box2_center_y)**2)
        
        # Consider the half-widths and half-heights
        box1_half_width = (box1[2] - box1[0]) / 2
        box1_half_height = (box1[3] - box1[1]) / 2
        box2_half_width = (box2[2] - box2[0]) / 2
        box2_half_height = (box2[3] - box2[1]) / 2

        # Calculate distance along x and y axes between centers
        dx = abs(box1_center_x - box2_center_x)
        dy = abs(box1_center_y - box2_center_y)

        # Calculate overlap or gap along x and y axes
        # This is the distance between the closest edges if they were aligned.
        gap_x = dx - (box1_half_width + box2_half_width)
        gap_y = dy - (box1_half_height + box2_half_height)

        # If both gaps are negative, boxes are overlapping (or one contains the other)
        if gap_x < 0 and gap_y < 0:
            return 0.0 # Overlapping

        # If one gap is negative and the other positive, they might still be "close"
        # but not overlapping in a simple way. The center distance minus radii is a good approximation.
        # For simplicity and to match the previous intent of "distance between centers minus radii sum":
        box1_avg_radius = (box1_half_width + box1_half_height) / 2 # (width+height)/4
        box2_avg_radius = (box2_half_width + box2_half_height) / 2

        proximity = center_distance - (box1_avg_radius + box2_avg_radius)
        return max(0, proximity) # Return 0 if "effective overlap" by this measure

    def _get_player_detections(self, tracked_detections: sv.Detections) -> sv.Detections:
        if not tracked_detections or len(tracked_detections.xyxy) == 0 or tracked_detections.class_id is None or tracked_detections.tracker_id is None:
            return sv.Detections.empty()
        
        player_entity_ids = [config.TEAM_A_ID, config.TEAM_B_ID, config.PLAYER_ID, config.GOALKEEPER_ID]
        player_mask = np.isin(tracked_detections.class_id, player_entity_ids)
        return tracked_detections[player_mask]

    def process_frame(self, frame_idx: int, annotated_frame: np.ndarray,
                      ball_detections: sv.Detections,
                      tracked_detections: sv.Detections):
        if not config.CLIP_EXTRACTION_ENABLED:
            return

        self.annotated_frame_buffer.append({'frame_idx': frame_idx, 'frame_data': annotated_frame.copy()})

        player_detections = self._get_player_detections(tracked_detections)
        current_interacting_player_ids = set()

        if not ball_detections or len(ball_detections.xyxy) == 0 or not player_detections or len(player_detections.xyxy) == 0:
            ended_interactions = list(self.active_interactions.keys())
            for player_id in ended_interactions:
                self._finalize_interaction(player_id, frame_idx, reason="no ball or no players detected")
            return

        ball_bbox = ball_detections.xyxy[0] 

        for i in range(len(player_detections.xyxy)):
            player_bbox = player_detections.xyxy[i]
            player_id = player_detections.tracker_id[i]

            if player_id is None:
                logging.warning(f"[Frame {frame_idx}] Player detection with None tracker_id. Skipping.")
                continue

            iou = self._calculate_iou(ball_bbox, player_bbox)
            proximity = self._calculate_proximity(ball_bbox, player_bbox)
            
            is_interacting = (iou >= self.interaction_iou_threshold) or \
                             (proximity <= self.proximity_threshold_pixels)

            if is_interacting:
                current_interacting_player_ids.add(player_id)
                if player_id not in self.active_interactions:
                    self.active_interactions[player_id] = {
                        'start_frame': frame_idx,
                        'last_active_frame': frame_idx,
                        'ball_bbox_at_start': ball_bbox.copy()
                    }
                    interaction_type = "IoU" if iou >= self.interaction_iou_threshold else "proximity"
                    interaction_value = iou if interaction_type == "IoU" else proximity
                    logging.debug(f"[Frame {frame_idx}] Interaction started: Ball and Player {player_id} "
                                  f"({interaction_type}: {interaction_value:.2f})")
                else:
                    self.active_interactions[player_id]['last_active_frame'] = frame_idx
            else: # Not interacting
                if player_id in self.active_interactions:
                    self._finalize_interaction(player_id, frame_idx, 
                                               reason=f"interaction ended (IoU: {iou:.2f}, Prox: {proximity:.2f}px)")
        
        detected_player_ids_in_frame = set(player_detections.tracker_id) if player_detections.tracker_id is not None else set()
        lost_or_stopped_interacting_players = set(self.active_interactions.keys()) - current_interacting_player_ids
        
        for player_id in lost_or_stopped_interacting_players:
            # If player track is lost OR if player is still detected but no longer meets interaction criteria
            # The latter case is handled by the 'else' block inside the loop above.
            # This part primarily handles players who are no longer detected at all.
            if player_id not in detected_player_ids_in_frame:
               self._finalize_interaction(player_id, frame_idx, reason="player track lost")


    def _finalize_interaction(self, player_id: int, current_frame_idx: int, reason: str = "unknown"):
        if player_id in self.active_interactions:
            interaction_data = self.active_interactions.pop(player_id)
            
            core_interaction_start_frame = interaction_data['start_frame']
            core_interaction_end_frame = interaction_data['last_active_frame']
            actual_interaction_duration_frames = core_interaction_end_frame - core_interaction_start_frame + 1

            logging.debug(f"[Frame {current_frame_idx}] Interaction ending for Player {player_id} (Reason: {reason}). "
                          f"Core Interaction: [{core_interaction_start_frame}-{core_interaction_end_frame}], "
                          f"Duration: {actual_interaction_duration_frames} frames.")

            if actual_interaction_duration_frames >= self.min_interaction_duration_frames:
                self.clip_event_counter += 1
                event_id = self.clip_event_counter

                # Define initial clip boundaries based on pre/post buffers around the core interaction
                tentative_clip_start_frame = max(0, core_interaction_start_frame - self.frames_pre_interaction_buffer)
                tentative_clip_end_frame = core_interaction_end_frame + self.frames_post_interaction_buffer
                
                tentative_clip_duration_frames = tentative_clip_end_frame - tentative_clip_start_frame + 1

                final_clip_start_frame = tentative_clip_start_frame
                final_clip_end_frame = tentative_clip_end_frame

                # Ensure minimum total clip length (e.g., 4 seconds)
                if tentative_clip_duration_frames < self.min_total_clip_frames:
                    frames_to_add = self.min_total_clip_frames - tentative_clip_duration_frames
                    # Add extra frames to the end of the clip to meet minimum length
                    final_clip_end_frame += frames_to_add
                    logging.debug(f"Clip for Player {player_id} extended by {frames_to_add} frames to meet min total duration.")
                
                base_filename = config.CLIP_FILENAME_TEMPLATE.format(
                    event_id=event_id,
                    player_id=player_id,
                    interaction_start_frame=core_interaction_start_frame # Filename uses core interaction start
                )
                base_filename = "".join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in base_filename)
                clip_output_path = os.path.join(self.output_dir, base_filename)

                logging.info(
                    f"CLIP DEFINED: Event {event_id}, Player {player_id}. "
                    f"Core Interaction: [{core_interaction_start_frame}-{core_interaction_end_frame}] ({actual_interaction_duration_frames}f). "
                    f"Final Clip Window: [{final_clip_start_frame}-{final_clip_end_frame}] ({(final_clip_end_frame - final_clip_start_frame + 1)/self.fps:.2f}s). "
                    f"Saving to {clip_output_path}"
                )
                self._write_clip(final_clip_start_frame, final_clip_end_frame, clip_output_path)
            else:
                logging.debug(f"Interaction for Player {player_id} (duration {actual_interaction_duration_frames}f) "
                              f"was too short. Min required: {self.min_interaction_duration_frames}f. Not saving clip.")

    def _write_clip(self, clip_start_frame: int, clip_end_frame: int, output_path: str):
        frames_for_clip = []
        
        # Log buffer state for diagnostics
        if self.annotated_frame_buffer:
            actual_earliest_buffered_idx = self.annotated_frame_buffer[0]['frame_idx']
            actual_latest_buffered_idx = self.annotated_frame_buffer[-1]['frame_idx']
            logging.debug(f"Writing clip {output_path}. Desired frames: [{clip_start_frame}-{clip_end_frame}]. "
                          f"Buffer has {len(self.annotated_frame_buffer)} frames (idx {actual_earliest_buffered_idx} to {actual_latest_buffered_idx})")
        else:
            logging.warning(f"Annotated frame buffer is empty when trying to write clip {output_path}.")
            return

        # Iterate over a copy of the deque for safety if it were to be modified elsewhere (though not expected here)
        for buffered_item in list(self.annotated_frame_buffer): 
            if buffered_item['frame_idx'] >= clip_start_frame and buffered_item['frame_idx'] <= clip_end_frame:
                frames_for_clip.append(buffered_item['frame_data'])
        
        if not frames_for_clip:
            logging.warning(f"No frames found in buffer for clip {output_path} (target frames {clip_start_frame}-{clip_end_frame}). "
                            f"Buffer might not cover the required range (oldest frame: {self.annotated_frame_buffer[0]['frame_idx'] if self.annotated_frame_buffer else 'N/A'}). Skipping clip.")
            return
        
        # Check if enough frames were retrieved relative to what was expected.
        expected_num_frames = clip_end_frame - clip_start_frame + 1
        if len(frames_for_clip) < expected_num_frames:
            logging.warning(f"Retrieved {len(frames_for_clip)} frames for clip {output_path}, but expected {expected_num_frames}. "
                            f"Clip might be shorter than intended due to buffer limitations for frames [{clip_start_frame}-{clip_end_frame}].")


        clip_fps = config.CLIP_FPS_RATE if config.CLIP_FPS_RATE is not None and config.CLIP_FPS_RATE > 0 else self.fps
        
        sample_frame_for_res = frames_for_clip[0]
        height, width, _ = sample_frame_for_res.shape
        clip_resolution = config.CLIP_RESOLUTION if config.CLIP_RESOLUTION else (width, height)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4 files
        video_writer = cv2.VideoWriter(output_path, fourcc, clip_fps, clip_resolution)

        if not video_writer.isOpened():
            logging.error(f"Could not open video writer for clip: {output_path}")
            return

        logging.info(f"Writing {len(frames_for_clip)} frames to clip: {output_path} (FPS: {clip_fps:.2f}, Res: {clip_resolution})")
        for frame_data in frames_for_clip:
            if frame_data.shape[1] != clip_resolution[0] or frame_data.shape[0] != clip_resolution[1]:
                frame_data_resized = cv2.resize(frame_data, clip_resolution, interpolation=cv2.INTER_AREA)
                video_writer.write(frame_data_resized)
            else:
                video_writer.write(frame_data)
        
        video_writer.release()
        logging.info(f"Successfully saved clip: {output_path}")

    def finalize_all_clips(self, last_processed_frame_idx: int):
        if not config.CLIP_EXTRACTION_ENABLED:
            return
            
        logging.info(f"Finalizing any remaining active interactions for clip extraction at frame {last_processed_frame_idx}...")
        active_player_ids = list(self.active_interactions.keys()) 
        for player_id in active_player_ids:
            self._finalize_interaction(player_id, last_processed_frame_idx, reason="video ended")
        logging.info(f"Clip finalization complete. {self.clip_event_counter} clip events processed in total.")
