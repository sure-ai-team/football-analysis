#!/usr/bin/env python3
import os
import argparse
import cv2
import numpy as np
import torch
import supervision as sv
from tqdm import tqdm
from pathlib import Path

# Import custom modules
from sports.common.team import TeamClassifier
from sports.common.detection import FootballDetector
from sports.common.visualize import FootballVisualizer

# Define the missing function
def resolve_goalkeepers_team_id(player_detections, goalkeeper_detections):
    """
    Assign team IDs to goalkeepers based on their proximity to field players.
    
    Args:
        player_detections: Detections object for field players with team IDs assigned
        goalkeeper_detections: Detections object for goalkeepers without team IDs
    
    Returns:
        Array of team IDs for goalkeepers
    """
    gk_team_ids = np.zeros(len(goalkeeper_detections), dtype=int)
    
    if len(player_detections.xyxy) == 0 or len(goalkeeper_detections.xyxy) == 0:
        return gk_team_ids
    
    # Get centroids of all detections
    player_centers = np.array([
        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        for box in player_detections.xyxy
    ])
    
    goalkeeper_centers = np.array([
        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        for box in goalkeeper_detections.xyxy
    ])
    
    # Get unique team IDs from players
    unique_team_ids = np.unique(player_detections.class_id)
    if len(unique_team_ids) < 2:
        return gk_team_ids
    
    # For each goalkeeper, find the closest group of players
    for i, gk_center in enumerate(goalkeeper_centers):
        team_distances = {}
        
        # Calculate average distance to players of each team
        for team_id in unique_team_ids:
            team_players = player_centers[player_detections.class_id == team_id]
            if len(team_players) == 0:
                continue
                
            # Calculate distances to all players of this team
            distances = np.sqrt(np.sum((team_players - gk_center) ** 2, axis=1))
            # Use the minimum distance to any player of this team
            team_distances[team_id] = np.min(distances)
        
        # Assign goalkeeper to team with largest average distance
        # (goalkeepers are typically far from their own team's players)
        if team_distances:
            gk_team_ids[i] = max(team_distances, key=team_distances.get)
    
    return gk_team_ids

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Football player tracking and team classification")
    
    parser.add_argument(
        "--source", "-s", type=str, required=True, help="Path to the source video"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Path for the output video"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="app/models/yolo11_football_v2/weights/best.pt",
        help="Path to the YOLO detection model"
    )
    parser.add_argument(
        "--reid", "-r", type=str, default="osnet_x0_25_msmt17.pt",
        help="Path to the ReID weights for BotSort tracker"
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)"
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.3,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--stride", type=int, default=30,
        help="Frame stride for team classification training"
    )
    
    return parser.parse_args()

def main():
    """Main function to run football tracking pipeline."""
    # Parse command-line arguments
    args = parse_args()
    
    # Configure CUDA execution providers if using GPU
    if args.device == "cuda":
        os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
    
    # Initialize detector with tracker
    print(f"Initializing detector with model: {args.model}")
    detector = FootballDetector(
        model_path=args.model,
        reid_weights_path=args.reid,
        device=args.device
    )
    
    # Initialize visualizer
    visualizer = FootballVisualizer()
    
    # Determine output path if not specified
    if args.output is None:
        source_path = Path(args.source)
        args.output = str(source_path.parent / f"{source_path.stem}-result.mp4")
    
    # Step 1: Collect player crops for team classification
    print(f"Collecting player crops from video for team classification...")
    player_crops = detector.collect_player_crops(args.source, args.stride)
    print(f"Collected {len(player_crops)} player crops")
    
    if len(player_crops) == 0:
        print("No player crops found. Cannot perform team classification.")
        return
    
    # Step 2: Initialize and fit team classifier
    print("Fitting team classifier...")
    team_classifier = TeamClassifier(device=args.device)
    team_classifier.fit(player_crops)
    print("Team classifier fitted successfully")
    
    # Step 3: Process full video
    print(f"Processing video: {args.source}")
    # Get video properties
    video_info = sv.VideoInfo.from_video_path(args.source)
    
    # Create video writer
    video_writer = cv2.VideoWriter(
        filename=args.output,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=video_info.fps,
        frameSize=(video_info.width, video_info.height)
    )
    
    # Process each frame
    frame_generator = sv.get_video_frames_generator(args.source)
    progress_bar = tqdm(total=video_info.total_frames, desc="Processing video")
    
    frame_count = 0
    for frame in frame_generator:
        # Process frame with detector and tracker
        ball_detections, player_detections, goalkeeper_detections, referee_detections = (
            detector.process_frame(frame, conf_threshold=args.conf_threshold)
        )
        
        # Classify players into teams if there are any players detected
        if len(player_detections.xyxy) > 0:
            player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
            player_detections.class_id = team_classifier.predict(player_crops)
        
        # Resolve goalkeeper team IDs if both players and goalkeepers are detected
        if len(player_detections.xyxy) > 0 and len(goalkeeper_detections.xyxy) > 0:
            goalkeeper_detections.class_id = resolve_goalkeepers_team_id(
                player_detections, goalkeeper_detections
            )
        
        # Visualize the frame
        annotated_frame = visualizer.annotate_frame(
            frame=frame,
            players_detections=player_detections,
            goalkeepers_detections=goalkeeper_detections,
            referees_detections=referee_detections,
            ball_detections=ball_detections
        )
        
        # Write the annotated frame
        video_writer.write(annotated_frame)
        
        # Update progress
        frame_count += 1
        progress_bar.update(1)
        if frame_count >= video_info.total_frames:
            break
    
    # Release resources
    video_writer.release()
    progress_bar.close()
    
    print(f"Video processing completed. Output saved to: {args.output}")

if __name__ == "__main__":
    main()