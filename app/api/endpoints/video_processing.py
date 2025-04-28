from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil
import logging
import traceback
import time
import torch # For checking GPU availability

from app.core import config
from app.processing.video_processor import VideoProcessor
from app.api.models import ProcessingResponse, HealthCheckResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# --- Background Task Function ---
def run_video_processing_task(temp_video_path: str, output_dir: str):
    """
    Function executed by the background task to process the video.
    Includes error handling.
    """
    logger.info(f"Background task started for video: {temp_video_path}")
    start_time = time.time()
    try:
        processor = VideoProcessor(source_video_path=temp_video_path, output_dir=output_dir)
        output_video, output_json = processor.run() # This blocks until done

        if output_video and output_json:
            logger.info(f"Background task completed successfully for: {temp_video_path}")
            logger.info(f"Output video: {output_video}")
            logger.info(f"Output JSON: {output_json}")
        else:
            logger.error(f"Background task failed to produce output for: {temp_video_path}")

    except FileNotFoundError as e:
        logger.error(f"Error in background task (FileNotFound): {e}", exc_info=True)
        # Handle file not found specifically - maybe the temp file was deleted?
    except Exception as e:
        logger.error(f"Unhandled error in background processing task for {temp_video_path}: {e}", exc_info=True)
        # Log the full traceback for debugging
        logger.error(traceback.format_exc())
    finally:
        # --- Cleanup ---
        # Optionally remove the temporary uploaded file after processing
        try:
            if config.TEMP_UPLOAD_DIR.exists() and temp_video_path:
                 temp_path_obj = Path(temp_video_path)
                 if temp_path_obj.exists() and temp_path_obj.is_relative_to(config.TEMP_UPLOAD_DIR):
                      temp_path_obj.unlink()
                      logger.info(f"Removed temporary upload file: {temp_video_path}")
                 else:
                      logger.warning(f"Skipping removal of non-temporary file: {temp_video_path}")

        except Exception as cleanup_e:
            logger.error(f"Error removing temporary file {temp_video_path}: {cleanup_e}")

        end_time = time.time()
        logger.info(f"Background task finished for {temp_video_path}. Total time: {end_time - start_time:.2f} seconds")


# --- API Endpoints ---

@router.post("/process-video/", response_model=ProcessingResponse, status_code=202)
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...)
    ):
    """
    Accepts a video file upload, saves it temporarily, and starts
    the processing pipeline in the background.
    Returns an immediate response indicating processing has started.
    """
    # Basic validation for file type (optional but recommended)
    allowed_content_types = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/avi"]
    if video_file.content_type not in allowed_content_types:
        logger.warning(f"Received unsupported file type: {video_file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {video_file.content_type}. Please upload MP4, MOV, or AVI."
        )

    # Ensure temporary upload directory exists
    config.TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Create a unique temporary file path
    # Using timestamp and original filename for uniqueness and traceability
    timestamp = int(time.time())
    temp_file_path = config.TEMP_UPLOAD_DIR / f"{timestamp}_{video_file.filename}"

    logger.info(f"Receiving video file: {video_file.filename} (Type: {video_file.content_type})")
    logger.info(f"Saving temporary file to: {temp_file_path}")

    try:
        # Save the uploaded file asynchronously
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        logger.info(f"Temporary file saved successfully: {temp_file_path}")

    except Exception as e:
        logger.error(f"Failed to save uploaded file {temp_file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")
    finally:
        # Ensure the file object is closed
        await video_file.close()


    # Add the processing task to background tasks
    # Pass the path to the *saved* temporary file
    try:
        background_tasks.add_task(
            run_video_processing_task,
            temp_video_path=str(temp_file_path), # Pass as string
            output_dir=str(config.DEFAULT_OUTPUT_DIR) # Pass output dir from config
        )
        logger.info(f"Video processing task added to background for: {temp_file_path}")

        # Return response indicating processing has started
        return ProcessingResponse(
            message="Video upload successful. Processing started in the background.",
            status="Processing started",
            # Optionally return expected output paths (though they don't exist yet)
            processed_video_path=str(config.DEFAULT_OUTPUT_DIR / f"{Path(video_file.filename).stem}_processed.mp4"),
            results_json_path=str(config.DEFAULT_OUTPUT_DIR / f"{Path(video_file.filename).stem}_results.json")
        )
    except Exception as e:
         logger.error(f"Failed to add background task for {temp_file_path}: {e}", exc_info=True)
         # Clean up the saved file if task scheduling fails
         if temp_file_path.exists():
             temp_file_path.unlink()
         raise HTTPException(status_code=500, detail=f"Could not start background processing task: {e}")


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Basic health check endpoint to verify service status and dependencies.
    """
    return HealthCheckResponse(
        status="OK",
        paddle_ocr_available=config.PADDLEOCR_AVAILABLE,
        gpu_available=torch.cuda.is_available(),
        reid_model_available=config.REID_WEIGHTS_PATH.exists() if config.TRACKER_WITH_REID else False
    )

# --- Placeholder for potentially getting results ---
# You might add another endpoint like GET /results/{task_id} or /results/{filename}
# to check the status or retrieve the paths once processing is done.
# This requires storing task status somewhere (e.g., database, in-memory dict).
# For simplicity, this example only starts the task and logs the output paths.
