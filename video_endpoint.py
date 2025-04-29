from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import tempfile
import shutil
import os
import uuid
import logging

# Import your existing main processing function
from main import main as process_video_main

app = FastAPI()

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    # Validate the uploaded file
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    # Create temporary files for input and output
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    try:
        # Save the uploaded file to the temporary input file
        with input_temp as f:
            shutil.copyfileobj(file.file, f)

        # Process the video using your existing main function
        process_video_main(input_temp.name, output_temp.name)

        # Return the processed video as a response
        return FileResponse(
            path=output_temp.name,
            media_type="video/mp4",
            filename=f"processed_{file.filename}"
        )
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during video processing.")
    finally:
        # Clean up the temporary input file
        try:
            os.unlink(input_temp.name)
        except Exception as e:
            logging.warning(f"Failed to delete temporary input file: {e}")
        # Note: The output file is returned in the response; deletion should be handled appropriately
