import os
import torch
import supervision as sv
from pathlib import Path

# --- Project Root ---
# Assuming this script is in app/core, the project root is two levels up.
# Adjust if the structure is different.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"

# --- Environment & Device ---
# Set ONNX runtime provider (do this early)
# os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]" # Or "[CPUExecutionProvider]"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Model Paths ---
YOLO_MODEL_PATH = APP_DIR / "models/yolo11_football_v2/weights/best.pt"
REID_WEIGHTS_PATH = APP_DIR / "models/clip_market1501.pt"
SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224' # Cloud model

# --- Input/Output Paths ---
DEFAULT_SOURCE_VIDEO_PATH = APP_DIR / "test_data/raw/121364_0.mp4"
DEFAULT_OUTPUT_DIR = APP_DIR / "test_data/processed"
OCR_DEBUG_DIR = APP_DIR / "ocr_debug_crops"

# Ensure output directories exist
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OCR_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
print(f"Default output directory: {DEFAULT_OUTPUT_DIR}")
print(f"OCR debug crops directory: {OCR_DEBUG_DIR}")

# --- Detection & Classification Configuration ---
YOLO_CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5
TEAM_CLASSIFIER_BATCH_SIZE = 64
TEAM_CLASSIFIER_STRIDE = 30 # Frame stride for collecting initial crops

# Class IDs (from initial detection model)
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# Team/Role Class IDs (assigned *after* classification)
TEAM_A_ID = 0 # Example ID for Team A (defends left goal in fallback)
TEAM_B_ID = 1 # Example ID for Team B (defends right goal in fallback)
REFEREE_TEAM_ID = 2 # Example ID for Referee

# Color Configuration
DEFAULT_TEAM_A_COLOR = sv.Color.from_hex('#FF0000') # Red
DEFAULT_TEAM_B_COLOR = sv.Color.from_hex('#00FFFF') # Yellow / Cyan in notebook? Let's stick to notebook comment: Yellow
DEFAULT_REFEREE_COLOR = sv.Color.from_hex('#FFFF00') # Cyan / Yellow in notebook? Let's stick to notebook comment: Cyan
FALLBACK_COLOR = sv.Color.from_hex('#808080') # Grey
COLOR_SIMILARITY_THRESHOLD = 50.0 # Max RGB distance diff to be considered ambiguous
CENTRAL_FRACTION_FOR_COLOR = 0.5 # Fraction of bbox center to use for avg color

# --- Tracking Configuration ---
TRACKER_HALF_PRECISION = False # Set based on device/performance needs
TRACKER_WITH_REID = REID_WEIGHTS_PATH.exists()

# --- OCR Configuration ---
# Attempt to import PaddleOCR and set flag
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    print("PaddleOCR found.")
except ImportError:
    print("Warning: PaddleOCR not found. OCR functionality will be disabled.")
    PADDLEOCR_AVAILABLE = False

OCR_USE_GPU = (DEVICE.type == 'cuda')
OCR_LANGUAGE = 'en'
OCR_CONFIDENCE_THRESHOLD = 0.8
MIN_JERSEY_DIGITS = 1
MAX_JERSEY_DIGITS = 2

# --- ID Management Configuration ---
LOST_TRACK_MEMORY_SECONDS = 20
MISMATCH_CONSISTENCY_FRAMES = 3

# --- Annotation Configuration ---
ELLIPSE_THICKNESS = 1
LABEL_TEXT_COLOR = sv.Color.BLACK
LABEL_TEXT_POSITION = sv.Position.BOTTOM_CENTER
LABEL_TEXT_SCALE = 0.4
LABEL_TEXT_THICKNESS = 1
BALL_TRAIL_SECONDS = 1
BALL_TRAIL_BASE_COLOR = (255, 255, 0) # Bright Cyan (BGR) - Note: BGR for OpenCV
BALL_TRAIL_THICKNESS = 1
SPARKLE_COUNT = 3
SPARKLE_RADIUS = 2
SPARKLE_OFFSET = 3
MAX_BALL_DISTANCE_PER_FRAME = 400 # Max pixels ball can move between frames (TUNE THIS VALUE!)
SPARKLE_BASE_INTENSITY = 150
SPARKLE_MAX_INTENSITY = 255
CURRENT_BALL_MARKER_RADIUS = 4
CURRENT_BALL_MARKER_COLOR = (255, 255, 255) # White (BGR)
CURRENT_BALL_MARKER_THICKNESS = -1 # Filled

# --- API Configuration ---
API_HOST = "0.0.0.0"
API_PORT = 8000
TEMP_UPLOAD_DIR = APP_DIR / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
import logging
logging.basicConfig(level=logging.WARNING) # Show only warnings and errors
logging.disable(logging.INFO) # Disable INFO messages specifically

# --- Warnings ---
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings('ignore', category=UserWarning, module='paddle')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

print("-" * 30)
print("Configuration loaded.")
print(f"Project Root: {PROJECT_ROOT}")
print(f"App Directory: {APP_DIR}")
print(f"YOLO Model: {'Exists' if YOLO_MODEL_PATH.exists() else 'Not Found'}")
print(f"ReID Weights: {'Exists' if REID_WEIGHTS_PATH.exists() else 'Not Found'} (ReID Enabled: {TRACKER_WITH_REID})")
print(f"PaddleOCR Available: {PADDLEOCR_AVAILABLE}")
print("-" * 30)
