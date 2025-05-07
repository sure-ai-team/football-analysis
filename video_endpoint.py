from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import tempfile
import shutil
import os
import uuid
import logging
import time
# Import your existing main processing function
from main import main as process_video_main

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI()


PROCESSED_VIDEOS_DIR = "processed_videos"

try:
    os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)
    logger.info(f"Ensured directory exists: {PROCESSED_VIDEOS_DIR}")
except OSError as e:
    logger.error(f"Could not create directory {PROCESSED_VIDEOS_DIR}: {e}")


@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    """
    Receives a video file, processes it, saves the result to the
    'processed_videos' directory, and returns the processed file.
    """
    # Validate the uploaded file type
    allowed_extensions = ('.mp4', '.avi', '.mov')
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        logger.warning(f"Unsupported file format uploaded: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )

    # Create a temporary file for the uploaded (input) video
    # Using a context manager ensures it's cleaned up automatically
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as input_temp_file:
            # Save the uploaded file content to the temporary file
            shutil.copyfileobj(file.file, input_temp_file)
            input_temp_path = input_temp_file.name # Get the path for processing
            logger.info(f"Uploaded file saved to temporary path: {input_temp_path}")

    except Exception as e:
        logger.error(f"Failed to save uploaded file to temporary storage: {e}")
        raise HTTPException(status_code=500, detail="Failed to store uploaded file.")
    finally:
        # Ensure the uploaded file object is closed
        await file.close()

    # --- Define Output Path ---
    # Construct the output filename (e.g., 'myvideo_processed.mp4')
    filename_base = os.path.splitext(file.filename)[0]
    output_filename = f"{filename_base}_processed{file_ext}"
    # Construct the full path within the designated directory
    output_path = os.path.join(PROCESSED_VIDEOS_DIR, output_filename)
    logger.info(f"Output path set to: {output_path}")

    try:
        # --- Process the Video ---
        # Call your actual video processing function
        logger.info(f"Starting video processing for {file.filename}...")
        processing_start_time = time.perf_counter() # Start timer
        process_video_main(input_temp_path, output_path)
        processing_end_time = time.perf_counter() # End timer
        processing_duration = processing_end_time - processing_start_time
        logger.info(f"Video processing for {file.filename} finished successfully in {processing_duration:.4f} seconds.")

        # --- Return the Processed File ---
        # Check if the output file was actually created by the process
        if not os.path.exists(output_path):
             logger.error(f"Processing completed but output file not found at: {output_path}")
             raise HTTPException(status_code=500, detail="Processing failed to produce an output file.")

        return FileResponse(
            path=output_path,
            media_type=f"video/{file_ext.lstrip('.')}", # Dynamically set media type
            filename=output_filename # Use the new filename for download
        )
    except HTTPException as e:
        # Re-raise HTTP exceptions directly
        raise e
    except Exception as e:
        # Log errors during the processing step
        logger.error(f"Error during video processing main function for {file.filename}: {e}", exc_info=True)
        # Log time elapsed before failure if timer started
        if processing_start_time:
            processing_fail_time = time.perf_counter()
            processing_duration_before_fail = processing_fail_time - processing_start_time
            logger.info(f"Processing for {file.filename} failed after {processing_duration_before_fail:.4f} seconds.")
        raise HTTPException(status_code=500, detail="Internal server error during video processing.")
    finally:
        # --- Clean Up Temporary Input File ---
        try:
            os.unlink(input_temp_path)
            logger.info(f"Cleaned up temporary input file: {input_temp_path}")
        except OSError as e:
            # Log if deletion fails, but don't crash the request
            logger.warning(f"Could not delete temporary input file {input_temp_path}: {e}")