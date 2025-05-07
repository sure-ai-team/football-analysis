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
PLAYER_DETECTION_MODEL_PATH = "app/models/yolo_7-5-2025.pt"
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

# --- Class IDs (from detection model) ---
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# --- Team/Role IDs (assigned *after* classification/resolution) ---
TEAM_A_ID = 0 # Example ID for Team A (defends left goal in fallback)
TEAM_B_ID = 1 # Example ID for Team B (defends right goal in fallback)
REFEREE_TEAM_ID = 2 # Example ID for Referee

# --- Color Configuration ---
DEFAULT_TEAM_A_COLOR = sv.Color.from_hex('#FF0000') # Red
DEFAULT_TEAM_B_COLOR = sv.Color.from_hex('#00FFFF') # Cyan (Original was Yellow, changed based on comment)
DEFAULT_REFEREE_COLOR = sv.Color.from_hex('#FFFF00') # Yellow (Original was Cyan, changed based on comment)
FALLBACK_COLOR = sv.Color.from_hex('#808080') # Grey
COLOR_SIMILARITY_THRESHOLD = 50.0 # Max RGB distance diff to be considered ambiguous for GK resolution
CENTRAL_FRACTION_FOR_COLOR = 0.5 # Fraction of bbox center to use for average color

# --- OCR Configuration ---
OCR_ENABLED = True # Set to False to disable OCR
OCR_DEBUG_DIR = "ocr_debug_crops" # Directory to save OCR debug crops
OCR_CONFIDENCE_THRESHOLD = 0.8
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
BALL_TRAIL_BASE_COLOR = (0, 255, 255) # Bright Cyan (BGR) - Adjusted from original comment
BALL_TRAIL_THICKNESS = 1
SPARKLE_BASE_INTENSITY = 150
SPARKLE_MAX_INTENSITY = 255
CURRENT_BALL_MARKER_RADIUS = 4
CURRENT_BALL_MARKER_COLOR = (255, 255, 255) # White (BGR)
CURRENT_BALL_MARKER_THICKNESS = -1 # Filled

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