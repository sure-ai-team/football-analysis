import numpy as np
import cv2
import logging
import os
import traceback

from app.core import config

# Configure logging
logger = logging.getLogger(__name__)

# Initialize PaddleOCR instance globally within the module if available
ocr_model = None
if config.PADDLEOCR_AVAILABLE:
    try:
        # Import here to avoid import error if not installed
        from paddleocr import PaddleOCR
        # Initialize PaddleOCR (models are downloaded automatically on first run)
        ocr_model = PaddleOCR(
            use_angle_cls=False, # Don't detect text angle
            lang=config.OCR_LANGUAGE,
            use_gpu=config.OCR_USE_GPU,
            show_log=False # Suppress PaddleOCR internal logs
        )
        logger.info(f"PaddleOCR initialized successfully (lang='{config.OCR_LANGUAGE}', use_gpu={config.OCR_USE_GPU}).")
    except ImportError:
        # This case should ideally be caught by config.PADDLEOCR_AVAILABLE,
        # but double-check.
        logger.warning("PaddleOCR import failed even though PADDLEOCR_AVAILABLE was True.")
        config.PADDLEOCR_AVAILABLE = False
        ocr_model = None
    except Exception as e:
        logger.error(f"Error initializing PaddleOCR: {e}", exc_info=True)
        # Ensure OCR is marked as unavailable if init fails
        config.PADDLEOCR_AVAILABLE = False
        ocr_model = None

def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray | None:
    """
    Applies basic preprocessing to an image crop for potentially better OCR results.

    Args:
        crop: The input image crop (BGR NumPy array).

    Returns:
        The preprocessed image (Grayscale NumPy array), or None if input is invalid.
    """
    if crop is None or crop.size == 0:
        logger.warning("Received invalid crop for preprocessing.")
        return None

    # 1. Convert to Grayscale
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # --- Optional Enhancements (Uncomment or add as needed) ---
    # # 2. Resizing (if numbers are consistently small/large)
    # scale_factor = 2.0 # Example: Double the size
    # width = int(gray_crop.shape[1] * scale_factor)
    # height = int(gray_crop.shape[0] * scale_factor)
    # dim = (width, height)
    # gray_crop = cv2.resize(gray_crop, dim, interpolation=cv2.INTER_LINEAR) # Or INTER_CUBIC

    # # 3. Thresholding (can help in some cases, but might lose info)
    # # _, gray_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # gray_crop = cv2.adaptiveThreshold(gray_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    # #                                   cv2.THRESH_BINARY, 11, 2)

    # # 4. Denoising
    # # gray_crop = cv2.fastNlMeansDenoising(gray_crop, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # # 5. Sharpening
    # kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    # gray_crop = cv2.filter2D(gray_crop, -1, kernel)
    # --- End Optional Enhancements ---

    return gray_crop

def perform_ocr_on_crop(
    crop: np.ndarray,
    frame_idx: int | None = None, # Optional: for debug saving
    track_id: int | None = None   # Optional: for debug saving
    ) -> tuple[str | None, float | None]:
    """
    Performs OCR on a given crop using the initialized PaddleOCR model,
    filters results for valid jersey numbers, and returns the best match.

    Args:
        crop: The input image crop (BGR NumPy array).
        frame_idx: Optional frame index for debug image filenames.
        track_id: Optional track ID for debug image filenames.


    Returns:
        A tuple containing:
        - str | None: The detected jersey number (as a string) or None if no valid
                      number is found above the confidence threshold.
        - float | None: The confidence score of the detected number, or None.
    """
    global ocr_model # Access the globally initialized model

    # Check if OCR is available and the model is loaded
    if not config.PADDLEOCR_AVAILABLE or ocr_model is None:
        # logger.debug("OCR not available or model not loaded.")
        return None, None

    # Check if the input crop is valid
    if crop is None or crop.size == 0:
        logger.warning("Received invalid crop for OCR.")
        return None, None

    # Preprocess the crop
    processed_crop = preprocess_for_ocr(crop)
    if processed_crop is None:
        logger.warning("Preprocessing failed for OCR crop.")
        return None, None

    # --- Optional: Save debug crops ---
    if config.OCR_DEBUG_DIR.exists() and frame_idx is not None and track_id is not None:
        try:
            # Ensure filenames are valid
            safe_frame_idx = max(0, frame_idx)
            safe_track_id = max(0, track_id)
            # Save original crop
            # player_filename = config.OCR_DEBUG_DIR / f"frame{safe_frame_idx}_track{safe_track_id}_orig.png"
            # cv2.imwrite(str(player_filename), crop)
            # Save preprocessed (grayscale) crop used for OCR
            ocr_input_filename = config.OCR_DEBUG_DIR / f"frame{safe_frame_idx}_track{safe_track_id}_ocr_in.png"
            cv2.imwrite(str(ocr_input_filename), processed_crop)
        except Exception as write_e:
            logger.warning(f"Error saving OCR debug crop for track {track_id}: {write_e}")
    # --- End Optional Debug Saving ---


    try:
        # Perform OCR inference
        # Note: PaddleOCR expects BGR format if using color, but we send grayscale
        result = ocr_model.ocr(processed_crop, cls=False) # cls=False disables angle classification

        best_num_str: str | None = None
        highest_conf: float = 0.0

        # Process results: PaddleOCR returns a list of lists,
        # e.g., [[[[box_coords], ('text', confidence)], ...]]
        if result and result[0]: # Check if result is not None and the first element exists
            for res_item in result[0]:
                # Expected structure: [box, (text, confidence)]
                if len(res_item) == 2 and isinstance(res_item[1], tuple) and len(res_item[1]) == 2:
                    text, confidence = res_item[1]

                    # Validate the detected text and confidence
                    if (isinstance(text, str) and
                        text.isdigit() and # Check if the string contains only digits
                        config.MIN_JERSEY_DIGITS <= len(text) <= config.MAX_JERSEY_DIGITS and
                        isinstance(confidence, (float, int)) and # Check confidence type
                        confidence > config.OCR_CONFIDENCE_THRESHOLD):

                        # logger.debug(f"OCR Candidate: '{text}' (Conf: {confidence:.2f})")
                        # Keep the number with the highest confidence
                        if confidence > highest_conf:
                            highest_conf = float(confidence)
                            best_num_str = text
                    # else:
                        # logger.debug(f"OCR Reject: '{text}' (Conf: {confidence}, Valid: {text.isdigit() if isinstance(text, str) else False}, Len OK: {config.MIN_JERSEY_DIGITS <= len(text) <= config.MAX_JERSEY_DIGITS if isinstance(text, str) else False})")


        if best_num_str is not None:
            # logger.debug(f"OCR Best Match: '{best_num_str}' (Conf: {highest_conf:.2f})")
            return best_num_str, highest_conf
        else:
            # logger.debug("No valid jersey number found by OCR.")
            return None, None

    except ImportError:
        # Handle cases where paddle might be missing during runtime if checks fail
        logger.error("PaddleOCR is not installed. Cannot perform OCR.")
        config.PADDLEOCR_AVAILABLE = False
        ocr_model = None
        return None, None
    except Exception as e:
        logger.error(f"Error during PaddleOCR inference: {e}\n{traceback.format_exc()}")
        # Optionally try to release resources or re-initialize on specific errors
        return None, None

# Example usage (optional, for testing)
if __name__ == "__main__":
    # Create a dummy image with a number
    test_image = np.zeros((100, 100, 3), dtype=np.uint8) + 200 # Light gray background
    cv2.putText(test_image, "23", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3) # Black text

    print(f"PaddleOCR Available: {config.PADDLEOCR_AVAILABLE}")

    if config.PADDLEOCR_AVAILABLE and ocr_model:
        print("\nPerforming OCR on test image...")
        number, confidence = perform_ocr_on_crop(test_image, frame_idx=0, track_id=999)

        if number is not None:
            print(f"Detected Number: {number}")
            print(f"Confidence: {confidence:.4f}")
        else:
            print("No valid number detected.")

        # Test with an empty image
        print("\nPerforming OCR on empty image...")
        number_empty, conf_empty = perform_ocr_on_crop(np.array([]))
        print(f"Result for empty image: Number={number_empty}, Confidence={conf_empty}")

        # Test with an image without numbers
        print("\nPerforming OCR on image without numbers...")
        no_number_image = np.zeros((50, 150, 3), dtype=np.uint8) + 50
        number_none, conf_none = perform_ocr_on_crop(no_number_image)
        print(f"Result for no-number image: Number={number_none}, Confidence={conf_none}")

    else:
        print("\nSkipping OCR test as PaddleOCR is not available or failed to initialize.")

    # Check if debug dir exists
    print(f"\nOCR Debug Directory: {config.OCR_DEBUG_DIR}")
    print(f"Debug Directory Exists: {config.OCR_DEBUG_DIR.exists()}")
