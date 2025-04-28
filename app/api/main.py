from fastapi import FastAPI
import logging
import sys
import os

# --- Ensure app directory is in Python path ---
# This allows imports like `from app.core import config` to work correctly
# when running uvicorn from the project root directory.
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End Path Setup ---


from app.api.endpoints import video_processing
from app.core import config # Now this import should work

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO, # Set desired logging level (INFO, DEBUG, WARNING, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Log to console
        # Optionally add logging to a file:
        # logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Video Processing API",
    description="API for detecting, tracking, and classifying objects in sports videos.",
    version="0.1.0"
)

# Include routers from endpoint modules
app.include_router(video_processing.router, prefix="/api", tags=["Video Processing"])

@app.on_event("startup")
async def startup_event():
    """Log application startup details."""
    logger.info("--- Video Processing API Starting Up ---")
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"PaddleOCR available: {config.PADDLEOCR_AVAILABLE}")
    logger.info(f"ReID enabled: {config.TRACKER_WITH_REID}")
    logger.info(f"Temporary upload directory: {config.TEMP_UPLOAD_DIR}")
    logger.info(f"Default output directory: {config.DEFAULT_OUTPUT_DIR}")
    logger.info("API Ready.")

@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown."""
    logger.info("--- Video Processing API Shutting Down ---")
    # Add any cleanup needed on shutdown here
    logger.info("Shutdown complete.")


# Root endpoint (optional)
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Video Processing API. Go to /docs for documentation."}

# --- Main execution block (for running directly, though uvicorn is preferred) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Running API directly using uvicorn...")
    uvicorn.run(
        "app.api.main:app", # Point to the FastAPI app instance
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True # Enable auto-reload for development
        # You might want `reload=False` in production
    )

# run `uvicorn app.api.main:app --reload`