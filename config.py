# config.py
"""
Configuration settings for the video processing pipeline.
"""
import os
import torch
import supervision as sv
from pathlib import Path

# --- Environment and Device ---
# Set ONNX execution provider if needed (usually done before importing ONNX libraries)
# os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
DEVICE = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
HOME = os.getcwd()

# --- Model Paths ---
# PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/12" # Example Roboflow ID
PLAYER_DETECTION_MODEL_PATH = "app/models/yolo11/yolo11_football_5-11-20256/weights/last.pt"
# "app/models/yolo11_football_v2/weights/best.pt" # Local YOLO path
SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
REID_WEIGHTS_PATH = Path('clip_market1501.pt') # Path for BoTSORT ReID weights

# --- Team Classifier Setup ---
TEAM_CLASSIFIER_BATCH_SIZE = 64
TEAM_CLASSIFIER_STRIDE = 30 # Frame stride for collecting initial crops

# --- Detection and Tracking ---
DETECTION_CONFIDENCE_THRESHOLD = 0.3
DETECTION_NMS_THRESHOLD = 0.5
TRACKER_HALF_PRECISION = False # Set to True if using half precision for BoTSORT
TRACKER_WITH_REID = REID_WEIGHTS_PATH.exists() # Enable ReID if weights file exists

# --- Class IDs (from NEW detection model data.yaml) ---
# nc: 6
# names: ['Ball', 'Goalkeeper', 'Main referee', 'Player', 'Side referee', 'Staff members']
BALL_ID = 0
GOALKEEPER_ID = 2
MAIN_REFEREE_ID = 3
PLAYER_ID = 1
# SIDE_REFEREE_ID = 4
# STAFF_MEMBERS_ID = 5
# Note: Old REFEREE_ID = 3 is removed. Main and Side referees are now distinct.

# --- Team/Role IDs (assigned *after* classification/resolution) ---
# These are logical IDs used internally after initial detection and role assignment.
# Their specific values (0, 1, 2) are distinct from the detection model's class IDs.
TEAM_A_ID = 0 # Example ID for Team A (defends left goal in fallback)
TEAM_B_ID = 1 # Example ID for Team B (defends right goal in fallback)
REFEREE_TEAM_ID = 2 # Example ID for all Referees (Main or Side) after they are grouped.

# --- Color Configuration ---
DEFAULT_TEAM_A_COLOR = sv.Color.from_hex('#FF0000') # Red
DEFAULT_TEAM_B_COLOR = sv.Color.from_hex('#00FFFF') # Cyan
DEFAULT_REFEREE_COLOR = sv.Color.from_hex('#FFFF00') # Yellow
FALLBACK_COLOR = sv.Color.from_hex('#808080') # Grey
COLOR_SIMILARITY_THRESHOLD = 50.0 # Max RGB distance diff to be considered ambiguous for GK resolution
CENTRAL_FRACTION_FOR_COLOR = 0.5 # Fraction of bbox center to use for average color

# --- OCR Configuration ---
OCR_ENABLED = True # Set to False to disable OCR
OCR_DEBUG_DIR = "ocr_debug_crops" # Directory to save OCR debug crops
OCR_CONFIDENCE_THRESHOLD = 0.6
MIN_JERSEY_DIGITS = 1
MAX_JERSEY_DIGITS = 2
PADDLEOCR_LANG = 'en'
PADDLEOCR_USE_ANGLE_CLS = False

# --- Player ID Management ---
LOST_TRACK_MEMORY_SECONDS = 20 # How long to remember a lost track ID with a jersey number
MISMATCH_CONSISTENCY_FRAMES = 3 # How many consecutive frames a different jersey number must be seen to switch

# --- Ball Trail Configuration ---
BALL_TRAIL_ENABLED = False
BALL_TRAIL_SECONDS = 1
SPARKLE_COUNT = 3
SPARKLE_RADIUS = 2
SPARKLE_OFFSET = 3
MAX_BALL_DISTANCE_PER_FRAME = 400 # Max pixels ball can move between frames (Tune this!)
BALL_TRAIL_BASE_COLOR = (0, 255, 255) # Bright Cyan (BGR)
BALL_TRAIL_THICKNESS = 1
SPARKLE_BASE_INTENSITY = 150
SPARKLE_MAX_INTENSITY = 255
CURRENT_BALL_MARKER_RADIUS = 4
CURRENT_BALL_MARKER_COLOR = (255, 255, 255) # White (BGR)
CURRENT_BALL_MARKER_THICKNESS = -1 # Filled

# --- Ball Circle Annotation ---
BALL_ANNOTATION_COLOR = sv.Color.WHITE # Color for the ball circle
BALL_CIRCLE_THICKNESS = 2
BALL_LABEL_ENABLED = True
BALL_LABEL_TEXT = "Ball"

# --- Annotation Parameters ---
ELLIPSE_THICKNESS = 1
LABEL_TEXT_COLOR = sv.Color.BLACK
LABEL_TEXT_POSITION = sv.Position.BOTTOM_CENTER
LABEL_TEXT_SCALE = 0.4
LABEL_TEXT_THICKNESS = 1

# --- Video Processing ---
FRAME_STRIDE = 1 # Process every frame for tracking

# --- Logging ---
LOG_LEVEL = "WARNING" # e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

# --- Clip Extraction Configuration ---
CLIP_EXTRACTION_ENABLED = True  # Master switch for this feature
CLIP_OUTPUT_DIR = "action_clips" # Directory to save extracted clips

# Defines the window around the *core interaction* to save.
# For "1 sec before, 1 sec during, 1 sec after" feel, set these to 1.0.
# The "during" part is the actual detected interaction.
CLIP_SECONDS_BEFORE_INTERACTION = 1.0 # Seconds of footage before interaction starts
CLIP_SECONDS_AFTER_INTERACTION = 1.0  # Seconds of footage after interaction ends

# Minimum IoU between ball and player to be considered an interaction.
INTERACTION_IOU_THRESHOLD = 0.01 # Lowered, as per original request

# Maximum distance (in pixels) between the closest edges of ball and player bounding boxes
# to consider them interacting, even if IoU is below threshold.
# Adjust this based on your video resolution and typical distances.
PROXIMITY_THRESHOLD_PIXELS = 75 # Increased from 50 for more leniency

# Minimum duration an *actual interaction* (IoU or proximity met) must last
# to qualify for triggering a clip.
MIN_INTERACTION_DURATION_SECONDS = 0.2 # e.g., 0.2 seconds

CLIP_FILENAME_TEMPLATE = "event_{event_id}_player_{player_id}_interaction_{interaction_start_frame}.mp4"

CLIP_FPS_RATE = None # e.g., 15 or None to use source FPS
CLIP_RESOLUTION = None # e.g., (1280, 720) or None to use source resolution
