# models.py
"""
Functions for loading the required machine learning models.
"""
import torch
from ultralytics import YOLO
from transformers import AutoProcessor, SiglipVisionModel
import config
import logging
import warnings

# Suppress specific warnings if necessary
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings('ignore', category=UserWarning, module='paddle')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# Configure logging based on config file
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.WARNING))
if config.LOG_LEVEL.upper() == "WARNING":
    logging.disable(logging.INFO) # Disable INFO messages specifically for WARNING level

def load_player_detection_model():
    """Loads the YOLO model for player detection."""
    try:
        logging.info(f"Loading Player Detection model from: {config.PLAYER_DETECTION_MODEL_PATH}")
        # Note: The original notebook set ONNX env var, but YOLO loads PyTorch directly here.
        # If ONNX export/inference is intended later, that needs separate handling.
        model = YOLO(config.PLAYER_DETECTION_MODEL_PATH)
        # Perform a dummy inference to ensure model is loaded onto the correct device
        # model.predict(torch.zeros(1, 3, 640, 640).to(config.DEVICE), verbose=False)
        logging.info(f"Player Detection model loaded successfully to device: {config.DEVICE}")
        return model
    except Exception as e:
        logging.error(f"Error loading Player Detection model: {e}", exc_info=True)
        raise

def load_siglip_models():
    """Loads the SigLIP model and processor for embeddings."""
    try:
        logging.info(f"Loading SigLIP model and processor from: {config.SIGLIP_MODEL_PATH}")
        embeddings_model = SiglipVisionModel.from_pretrained(config.SIGLIP_MODEL_PATH).to(config.DEVICE)
        embeddings_processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL_PATH)
        logging.info("SigLIP model and processor loaded successfully.")
        return embeddings_model, embeddings_processor
    except Exception as e:
        logging.error(f"Error loading SigLIP models: {e}", exc_info=True)
        raise

def load_ocr_model():
    """Loads the PaddleOCR model if enabled and available."""
    if not config.OCR_ENABLED:
        logging.warning("OCR is disabled in the configuration.")
        return None, False # Return None model and False availability flag

    try:
        from paddleocr import PaddleOCR
        PADDLEOCR_AVAILABLE = True
    except ImportError:
        logging.warning("PaddleOCR not found. Please install it (`pip install paddlepaddle paddleocr`). OCR functionality will be disabled.")
        return None, False # Return None model and False availability flag

    try:
        logging.info("Initializing PaddleOCR...")
        ocr_model = PaddleOCR(
            use_angle_cls=config.PADDLEOCR_USE_ANGLE_CLS,
            lang=config.PADDLEOCR_LANG,
            use_gpu=(config.DEVICE.type == 'cuda'),
            show_log=(config.LOG_LEVEL.upper() in ["DEBUG", "INFO"]) # Show logs only if debug/info
        )
        logging.info("PaddleOCR initialized successfully.")
        return ocr_model, True
    except Exception as e:
        logging.error(f"Error initializing PaddleOCR: {e}. Disabling OCR.", exc_info=True)
        return None, False # Return None model and False availability flag

# Example usage (optional, for testing)
if __name__ == "__main__":
    print(f"Using device: {config.DEVICE}")
    try:
        yolo_model = load_player_detection_model()
        print("YOLO Model loaded.")
        # siglip_model, siglip_processor = load_siglip_models()
        # print("SigLIP Model and Processor loaded.")
        ocr_model, ocr_available = load_ocr_model()
        if ocr_available:
            print("OCR Model loaded.")
        else:
            print("OCR Model not loaded or unavailable.")
    except Exception as e:
        print(f"An error occurred during model loading test: {e}")

