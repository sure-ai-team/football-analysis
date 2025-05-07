# main.py
"""
Main script to run the football tracking and analysis pipeline.

Parses command-line arguments for input and output video paths,
initializes models, sets up the team classifier, processes the video
frame by frame, and saves the annotated output video.
"""
import argparse
import os
import sys
import logging
import traceback
from tqdm import tqdm
import cv2
import supervision as sv
import torch

# Import from local modules
import config
from models import load_player_detection_model, load_ocr_model
from team_classifier_setup import setup_team_classifier, TEAM_CLASSIFIER_AVAILABLE
from video_processor import initialize_tracker, initialize_ball_trail, process_frame

# Configure logging based on config file
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.WARNING),
                    format='%(asctime)s - %(levelname)s - %(message)s')
if config.LOG_LEVEL.upper() == "WARNING":
    logging.disable(logging.INFO) # Disable INFO messages specifically for WARNING level


def main(input_video_path: str, output_video_path: str):
    """
    Main function to execute the video processing pipeline.

    Args:
        input_video_path: Path to the source video file.
        output_video_path: Path where the annotated video will be saved.
    """
    logging.info(f"Starting video processing pipeline.")
    logging.info(f"Input video: {input_video_path}")
    logging.info(f"Output video: {output_video_path}")
    logging.info(f"Using device: {config.DEVICE}")

    # --- Argument Validation ---
    if not os.path.exists(input_video_path):
        logging.error(f"Input video path does not exist: {input_video_path}")
        sys.exit(1) # Exit if input video not found

    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logging.error(f"Could not create output directory '{output_dir}': {e}")
            sys.exit(1)

    if config.OCR_ENABLED and config.OCR_DEBUG_DIR and not os.path.exists(config.OCR_DEBUG_DIR):
         try:
             os.makedirs(config.OCR_DEBUG_DIR)
             logging.info(f"Created OCR debug directory: {config.OCR_DEBUG_DIR}")
         except OSError as e:
             logging.warning(f"Could not create OCR debug directory '{config.OCR_DEBUG_DIR}': {e}. Debug images will not be saved.")


    # --- Load Models ---
    try:
        player_detection_model = load_player_detection_model()
        # Note: SigLIP model is not directly used in the main loop based on the refactored code,
        # it was used in the initial clustering part which seems replaced by TeamClassifier.
        # If needed, uncomment:
        # siglip_model, siglip_processor = load_siglip_models()
        ocr_model, ocr_available = load_ocr_model()
    except Exception as e:
        logging.error(f"Failed to load models: {e}", exc_info=True)
        sys.exit(1) # Exit if models can't be loaded

    # --- Setup Team Classifier ---
    team_classifier = None
    if TEAM_CLASSIFIER_AVAILABLE: # Check if the class could be imported
        try:
            # Pass the detection model and video path for crop collection
            team_classifier = setup_team_classifier(player_detection_model, input_video_path)
            if team_classifier:
                logging.info("Team classifier setup completed successfully.")
            else:
                logging.warning("Team classifier setup failed or returned None. Player classification might be skipped or limited.")
        except Exception as e:
            logging.error(f"An error occurred during team classifier setup: {e}", exc_info=True)
            # Continue without team classifier if setup fails
            team_classifier = None
    else:
        logging.warning("TeamClassifier class not available. Skipping team classification setup.")


    # --- Initialize Tracker ---
    tracker = initialize_tracker()
    if tracker is None:
        logging.error("Failed to initialize tracker. Exiting.")
        sys.exit(1)

    # --- Video I/O Setup ---
    video_info = None
    fps = 30 # Default FPS if info retrieval fails
    total_frames = None
    try:
        logging.info("Getting video information...")
        video_info = sv.VideoInfo.from_video_path(str(input_video_path))
        width, height = video_info.width, video_info.height
        if video_info.fps and video_info.fps > 0:
            fps = video_info.fps
        else:
             logging.warning(f"Could not determine FPS from video info. Using default: {fps}")
        total_frames = video_info.total_frames if video_info.total_frames and video_info.total_frames > 0 else None
        logging.info(f"Video Info: {width}x{height}, FPS: {fps:.2f}, Total Frames: {total_frames if total_frames else 'Unknown'}")
    except Exception as e:
        logging.warning(f"Could not get video info using supervision: {e}. Attempting fallback with OpenCV.")
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            logging.error(f"Cannot open video file using OpenCV fallback: {input_video_path}")
            sys.exit(1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv_fps = cap.get(cv2.CAP_PROP_FPS)
        cv_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if cv_fps and cv_fps > 0: fps = cv_fps
        if cv_total_frames and cv_total_frames > 0: total_frames = cv_total_frames
        # Create a basic VideoInfo object for consistency if fallback is used
        video_info = sv.VideoInfo(width=width, height=height, fps=fps, total_frames=total_frames)
        logging.info(f"Fallback Video Info: {width}x{height}, FPS: {fps:.2f}, Total Frames: {total_frames if total_frames else 'Unknown'}")


    # Initialize Ball Trail Deque (needs FPS)
    # initialize_ball_trail(fps)

    # Initialize Video Reader and Writer
    frame_generator = sv.get_video_frames_generator(source_path=str(input_video_path), stride=config.FRAME_STRIDE)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
         logging.error(f"Could not open video writer for path: {output_video_path}")
         sys.exit(1)

    logging.info("Starting video processing loop...")
    # --- Main Video Processing Loop ---
    try:
        with tqdm(total=total_frames, desc="Processing video", unit="frame", disable=(total_frames is None)) as pbar:
            for frame_idx, frame in enumerate(frame_generator):
                if frame is None:
                    logging.warning(f"Received None frame at index {frame_idx}, ending processing.")
                    break

                # Calculate actual frame index if stride is not 1
                actual_frame_idx = frame_idx * config.FRAME_STRIDE

                try:
                    # Process the frame using the dedicated function
                    annotated_frame = process_frame(
                        frame=frame,
                        frame_idx=actual_frame_idx,
                        player_detection_model=player_detection_model,
                        team_classifier=team_classifier,
                        tracker=tracker,
                        ocr_model=ocr_model,
                        ocr_available=ocr_available,
                        video_info=video_info,
                        fps=fps
                    )

                    # Write the annotated frame to the output video
                    if annotated_frame is not None:
                        video_writer.write(annotated_frame)
                    else:
                        # Handle case where process_frame might return None unexpectedly
                        logging.error(f"process_frame returned None for frame {actual_frame_idx}. Writing original frame.")
                        video_writer.write(frame) # Write original frame as fallback

                except Exception as e_proc:
                    # Log critical errors during frame processing but try to continue
                    logging.error(f"--- CRITICAL ERROR processing frame {actual_frame_idx}: {e_proc} ---")
                    logging.error(traceback.format_exc())
                    logging.warning("Attempting to continue processing next frame. Writing original frame for current.")
                    # Write the original frame if processing failed catastrophically
                    video_writer.write(frame)

                if total_frames: pbar.update(1) # Update progress bar

    except KeyboardInterrupt:
        logging.warning("\nProcessing interrupted by user.")
    except Exception as e_main:
        logging.error(f"\n--- UNHANDLED EXCEPTION in main processing loop: {e_main} ---")
        logging.error(traceback.format_exc())
    finally:
        # --- Cleanup ---
        logging.info("Releasing video writer...")
        video_writer.release()
        # Close frame generator if it has a close method (good practice)
        if hasattr(frame_generator, 'close'):
             frame_generator.close()
        logging.info(f"Finished processing. Annotated video saved to: {output_video_path}")

        # Clear CUDA cache if GPU was used
        if config.DEVICE.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                logging.info("CUDA cache cleared.")
            except Exception as e_cuda:
                logging.warning(f"Error clearing CUDA cache: {e_cuda}")

        logging.info("Pipeline finished.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Football Player Tracking and Analysis Pipeline")
    parser.add_argument(
        "input_video",
        type=str,
        help="Path to the input video file."
    )
    parser.add_argument(
        "output_video",
        type=str,
        help="Path to save the annotated output video file."
    )
    args = parser.parse_args()

    # --- Run Main Function ---
    main(args.input_video, args.output_video)
