# action_extractor.py
"""
Module for detecting and extracting action clips from a football video.
This includes detection of ball-player interactions and extraction of 
relevant video segments for further action recognition processing.
"""
import cv2
import numpy as np
import supervision as sv
import os
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Deque
import time
import config

@dataclass
class InteractionEvent:
    """Class to track a single ball-player interaction event."""
    start_frame: int                   # First frame where interaction was detected
    player_ids: List[int] = field(default_factory=list)  # List of player IDs involved
    consecutive_frames: int = 0        # Count of consecutive frames with interaction
    last_frame: int = 0                # Last frame where interaction was seen
    is_active: bool = True             # Whether this event is still ongoing
    processed: bool = False            # Whether a clip has been extracted for this event
    interaction_frames: List[int] = field(default_factory=list)  # Frames with actual interaction

@dataclass
class ClipInfo:
    """Class to store information about an extracted action clip."""
    clip_id: str                      # Unique identifier for the clip
    start_frame: int                  # First frame of the clip
    end_frame: int                    # Last frame of the clip
    interaction_frames: List[int]     # Frames where ball-player interaction occurred
    player_ids: List[int]             # Player IDs involved in the interaction
    output_path: str                  # Path where the clip was saved

class ActionClipExtractor:
    """
    Class to detect ball-player interactions and extract action clips.
    """
    def __init__(self, video_info: sv.VideoInfo, fps: float):
        """
        Initialize the ActionClipExtractor.

        Args:
            video_info: Information about the source video
            fps: Frames per second of the source video
        """
        self.video_info = video_info
        self.fps = fps
        self.active_interactions: List[InteractionEvent] = []
        self.completed_interactions: List[InteractionEvent] = []
        self.extracted_clips: List[ClipInfo] = []
        self.frame_buffer: Deque[Tuple[int, np.ndarray]] = deque(maxlen=self._calculate_buffer_size())
        self.clip_writer = None
        self.current_clip_frames = []
        self.current_extraction_id = None
        
        # Create output directory if it doesn't exist
        if config.ENABLE_ACTION_CLIP_EXTRACTION and not os.path.exists(config.ACTION_CLIP_OUTPUT_DIR):
            try:
                os.makedirs(config.ACTION_CLIP_OUTPUT_DIR)
                logging.info(f"Created action clip output directory: {config.ACTION_CLIP_OUTPUT_DIR}")
            except OSError as e:
                logging.error(f"Failed to create action clip directory: {e}")
                # Continue without extraction if directory creation fails
    
    def _calculate_buffer_size(self) -> int:
        """Calculate the required buffer size based on clip duration and FPS."""
        if self.fps <= 0:
            # Default to 120 frames (4 seconds at 30fps) if FPS is invalid
            return 120
        
        # Calculate max frames needed for a clip (total duration + some extra buffer)
        return int((config.ACTION_CLIP_DURATION_SECONDS + 1.0) * self.fps)
    
    def _get_buffer_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Retrieve a frame from the buffer by its index."""
        for buffered_idx, frame in self.frame_buffer:
            if buffered_idx == frame_idx:
                return frame
        return None
    
    def _intersect_boxes(self, box1: np.ndarray, box2: np.ndarray) -> bool:
        """Check if two bounding boxes intersect."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Check if one box is to the left of the other
        if x1_max < x2_min or x2_max < x1_min:
            return False
        
        # Check if one box is above the other
        if y1_max < y2_min or y2_max < y1_min:
            return False
        
        # If we get here, the boxes overlap
        return True
    
    def detect_interactions(self, 
                           frame_idx: int, 
                           annotated_frame: np.ndarray, 
                           ball_detections: sv.Detections, 
                           people_detections: sv.Detections) -> bool:
        """
        Detect interactions between ball and players in the current frame.
        
        Args:
            frame_idx: Current frame index
            annotated_frame: The current annotated frame
            ball_detections: Ball detections in the current frame
            people_detections: Player/goalkeeper/referee detections in the current frame
            
        Returns:
            bool: Whether an ongoing extraction was completed in this call
        """
        if not config.ENABLE_ACTION_CLIP_EXTRACTION:
            return False
        
        # Add the current frame to the buffer
        self.frame_buffer.append((frame_idx, annotated_frame.copy()))
        
        # No ball or people detected, nothing to do
        if len(ball_detections) == 0 or len(people_detections) == 0:
            # Update active interactions as having no interactions this frame
            for interaction in self.active_interactions:
                interaction.consecutive_frames = 0
                if frame_idx - interaction.last_frame > int(self.fps * 1.0):  # If 1 second passed without interaction
                    interaction.is_active = False
                    self.completed_interactions.append(interaction)
            
            self.active_interactions = [i for i in self.active_interactions if i.is_active]
            return self._check_for_completed_extraction(frame_idx)
        
        # Detect interactions for this frame
        current_interactions = []
        has_interactions = False
        
        # Check each ball-player pair for intersection
        for ball_idx in range(len(ball_detections)):
            ball_box = ball_detections.xyxy[ball_idx]
            
            for person_idx in range(len(people_detections)):
                person_box = people_detections.xyxy[person_idx]
                
                if self._intersect_boxes(ball_box, person_box):
                    has_interactions = True
                    # Get the tracked ID of the player if available
                    player_id = people_detections.tracker_id[person_idx] if hasattr(people_detections, 'tracker_id') else person_idx
                    current_interactions.append(player_id)
        
        # Update active interactions
        if has_interactions:
            # Check if any existing active interaction matches current players
            interaction_updated = False
            
            for interaction in self.active_interactions:
                # Simple heuristic: If any player from current interaction is in this active interaction
                if any(p_id in interaction.player_ids for p_id in current_interactions):
                    interaction.consecutive_frames += 1
                    interaction.last_frame = frame_idx
                    # Add any new players not already tracked
                    for p_id in current_interactions:
                        if p_id not in interaction.player_ids:
                            interaction.player_ids.append(p_id)
                    interaction.interaction_frames.append(frame_idx)
                    interaction_updated = True
                    break
            
            # If no existing interaction was updated, create a new one
            if not interaction_updated:
                new_interaction = InteractionEvent(
                    start_frame=frame_idx,
                    player_ids=current_interactions.copy(),
                    consecutive_frames=1,
                    last_frame=frame_idx,
                    interaction_frames=[frame_idx]
                )
                self.active_interactions.append(new_interaction)
        else:
            # No interactions in this frame, decrease consecutive count for active interactions
            for interaction in self.active_interactions:
                interaction.consecutive_frames = 0
                # If no interaction for more than N frames, mark as inactive
                if frame_idx - interaction.last_frame > int(self.fps * 1.0):  # 1 second without interaction
                    interaction.is_active = False
                    self.completed_interactions.append(interaction)
        
        # Remove inactive interactions from active list
        self.active_interactions = [i for i in self.active_interactions if i.is_active]
        
        # Check if we need to extract a clip
        return self._process_for_extraction(frame_idx)
    
    def _process_for_extraction(self, current_frame_idx: int) -> bool:
        """Check if any interaction is ready for clip extraction."""
        # If already extracting, don't start another extraction
        if self.clip_writer is not None:
            return False
        
        extraction_completed = False
        
        # Check completed interactions first
        for interaction in self.completed_interactions:
            if interaction.processed:
                continue
                
            if max(interaction.consecutive_frames) >= config.MIN_INTERACTION_FRAMES_FOR_CLIP:
                self._start_clip_extraction(interaction, current_frame_idx)
                interaction.processed = True
                return False  # No completion yet
        
        # Then check active interactions
        for interaction in self.active_interactions:
            if interaction.processed:
                continue
                
            if interaction.consecutive_frames >= config.MIN_INTERACTION_FRAMES_FOR_CLIP:
                self._start_clip_extraction(interaction, current_frame_idx)
                interaction.processed = True
                return False  # No completion yet
        
        # Check if we need to finalize an in-progress extraction
        extraction_completed = self._check_for_completed_extraction(current_frame_idx)
        return extraction_completed
    
    def _start_clip_extraction(self, interaction: InteractionEvent, current_frame_idx: int):
        """Start extracting a clip for the given interaction."""
        earliest_interaction_frame = min(interaction.interaction_frames)
        
        # Calculate start frame with lead-in time
        lead_in_frames = int(config.ACTION_CLIP_LEAD_IN_SECONDS * self.fps)
        start_frame = max(0, earliest_interaction_frame - lead_in_frames)
        
        # Calculate end frame to achieve desired clip duration
        clip_duration_frames = int(config.ACTION_CLIP_DURATION_SECONDS * self.fps)
        end_frame = start_frame + clip_duration_frames
        
        # Adjust if we haven't reached the end frame yet in the video
        if end_frame > current_frame_idx:
            # We're still processing frames before the desired end frame
            # We'll continue collecting frames until we reach the end frame
            self.current_extraction_id = f"clip_{int(time.time())}_{start_frame}_{earliest_interaction_frame}"
            self.current_clip_frames = []
            
            # Set up the video writer
            output_path = os.path.join(config.ACTION_CLIP_OUTPUT_DIR, f"{self.current_extraction_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps_to_use = config.ACTION_CLIP_FPS_TARGET if config.ACTION_CLIP_FPS_TARGET > 0 else self.fps
            self.clip_writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                fps_to_use, 
                (self.video_info.width, self.video_info.height)
            )
            
            if not self.clip_writer.isOpened():
                logging.error(f"Failed to open clip writer for {output_path}")
                self.clip_writer = None
                return
            
            # Start adding frames from the buffer to the clip
            for buffered_idx, frame in self.frame_buffer:
                if start_frame <= buffered_idx <= current_frame_idx:
                    self.current_clip_frames.append(buffered_idx)
                    self.clip_writer.write(frame)
            
            # Record what we're extracting
            logging.info(f"Started extracting action clip {self.current_extraction_id} from frame {start_frame}")
            
            # Create clip info record
            self.extracted_clips.append(ClipInfo(
                clip_id=self.current_extraction_id,
                start_frame=start_frame,
                end_frame=end_frame,  # Target end frame
                interaction_frames=interaction.interaction_frames.copy(),
                player_ids=interaction.player_ids.copy(),
                output_path=output_path
            ))
    
    def _check_for_completed_extraction(self, current_frame_idx: int) -> bool:
        """Check if an in-progress extraction should be completed."""
        if self.clip_writer is None or not self.current_clip_frames:
            return False
        
        # Find the corresponding clip info
        clip_info = next((clip for clip in self.extracted_clips if clip.clip_id == self.current_extraction_id), None)
        if clip_info is None:
            return False
        
        # If we've reached the target end frame or we're close enough to the end of the video
        if current_frame_idx >= clip_info.end_frame:
            # Write any remaining frames
            for buffered_idx, frame in self.frame_buffer:
                if buffered_idx > max(self.current_clip_frames) and buffered_idx <= clip_info.end_frame:
                    self.clip_writer.write(frame)
                    self.current_clip_frames.append(buffered_idx)
            
            # Close the writer and clean up
            self.clip_writer.release()
            self.clip_writer = None
            
            logging.info(f"Completed extracting action clip {self.current_extraction_id}, "
                         f"{len(self.current_clip_frames)} frames written to {clip_info.output_path}")
            
            self.current_clip_frames = []
            self.current_extraction_id = None
            return True
        
        # If we need to add new frames as they come
        for buffered_idx, frame in self.frame_buffer:
            if buffered_idx > max(self.current_clip_frames) and buffered_idx <= clip_info.end_frame:
                self.clip_writer.write(frame)
                self.current_clip_frames.append(buffered_idx)
        
        return False
    
    def cleanup(self):
        """Clean up any resources used by the extractor."""
        if self.clip_writer is not None:
            self.clip_writer.release()
            logging.info(f"Released clip writer during cleanup.") 