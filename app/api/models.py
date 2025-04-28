from pydantic import BaseModel
from typing import Optional

class ProcessingRequest(BaseModel):
    """
    Model for the request (although the video comes as a file upload,
    we might add other parameters later). Currently unused but good practice.
    """
    # Example: Add options like forcing re-training, output resolution etc.
    # force_retrain: bool = False
    pass

class ProcessingResponse(BaseModel):
    """
    Model for the response sent back after processing.
    """
    message: str
    status: str # e.g., "Processing started", "Success", "Failed"
    processed_video_path: Optional[str] = None # Path relative to some base or absolute
    results_json_path: Optional[str] = None # Path to the JSON summary
    error_details: Optional[str] = None # Details if status is "Failed"

class HealthCheckResponse(BaseModel):
    """
    Model for the health check endpoint response.
    """
    status: str = "OK"
    paddle_ocr_available: bool
    gpu_available: bool
    reid_model_available: bool
