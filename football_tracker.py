import os
import cv2
import math
import random
import logging
import warnings
import traceback
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import torch
import umap
from sklearn.cluster import KMeans
from ultralytics import YOLO
from transformers import AutoProcessor, SiglipVisionModel
import supervision as sv
from more_itertools import chunked
from tqdm import tqdm

# Optional OCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    print("Warning: PaddleOCR not found. OCR functionality disabled.")
    PADDLEOCR_AVAILABLE = False

# Tracker
from boxmot import BotSort

# Environment
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

# Global Models (loaded once)
DEVICE = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

# Player detection
PLAYER_DETECTION_MODEL = YOLO("app/models/yolo11_football_v2/weights/best.pt")

# Embedding model
SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)

# OCR model
ocr_model = None
if PADDLEOCR_AVAILABLE:
    try:
        ocr_model = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=(DEVICE.type=='cuda'), show_log=False)
    except Exception:
        PADDLEOCR_AVAILABLE = False

# Logger & warnings
logging.basicConfig(level=logging.WARNING)
logging.disable(logging.INFO)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Constants
BATCH_SIZE = 64
STRIDE = 30
OCR_CONFIDENCE_THRESHOLD = 0.8
MIN_JERSEY_DIGITS = 1
MAX_JERSEY_DIGITS = 2

# Color defaults
DEFAULT_TEAM_A_COLOR = sv.Color.from_hex('#FF0000')
DEFAULT_TEAM_B_COLOR = sv.Color.from_hex('#00FFFF')
DEFAULT_REFEREE_COLOR = sv.Color.from_hex('#FFFF00')
FALLBACK_COLOR = sv.Color.from_hex('#808080')
COLOR_SIMILARITY_THRESHOLD = 50.0
TEAM_A_ID = 0
TEAM_B_ID = 1
REFEREE_TEAM_ID = 2

# IDs
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# Tracking
REID_WEIGHTS_PATH = Path('clip_market1501.pt')

# ----- Annotation Parameters -----
ELLIPSE_THICKNESS = 1
LABEL_TEXT_COLOR = sv.Color.BLACK
LABEL_TEXT_POSITION = sv.Position.BOTTOM_CENTER
LABEL_TEXT_SCALE = 0.4
LABEL_TEXT_THICKNESS = 1
BALL_TRAIL_BASE_COLOR = (255, 255, 0) # Bright Cyan (BGR)
BALL_TRAIL_THICKNESS = 1
SPARKLE_BASE_INTENSITY = 150
SPARKLE_MAX_INTENSITY = 255
CURRENT_BALL_MARKER_RADIUS = 4
CURRENT_BALL_MARKER_COLOR = (255, 255, 255) # White (BGR)
CURRENT_BALL_MARKER_THICKNESS = -1 # Filled

# OCR Configuration
OCR_CONFIDENCE_THRESHOLD = 0.8
MIN_JERSEY_DIGITS = 1
MAX_JERSEY_DIGITS = 2

# ID Management Configuration
LOST_TRACK_MEMORY_SECONDS = 20
MISMATCH_CONSISTENCY_FRAMES = 3

# Ball Trail Configuration
BALL_TRAIL_SECONDS = 1
SPARKLE_COUNT = 3
SPARKLE_RADIUS = 2
SPARKLE_OFFSET = 3
MAX_BALL_DISTANCE_PER_FRAME = 400 # Max pixels ball can move between frames (TUNE THIS VALUE!)


# Processing functions
def calculate_average_color(frame: np.ndarray, detections: sv.Detections, central_fraction: float = 0.5) -> sv.Color | None:
    if len(detections) == 0:
        return None
    avg_colors = []
    h, w, _ = frame.shape
    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x1 >= x2 or y1 >= y2:
            continue
        bw, bh = x2-x1, y2-y1
        cx, cy = x1 + bw//2, y1 + bh//2
        cw, ch = int(bw*central_fraction), int(bh*central_fraction)
        cx1 = max(x1, cx-cw//2)
        cy1 = max(y1, cy-ch//2)
        cx2 = min(x2, cx+cw//2)
        cy2 = min(y2, cy+ch//2)
        if cx1>=cx2 or cy1>=cy2:
            continue
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size>0:
            avg_colors.append(cv2.mean(crop)[:3])
    if not avg_colors:
        return None
    b, g, r = map(int, np.mean(avg_colors, axis=0))
    if r<50 and g<50 and b<50:
        r=g=b=50
    return sv.Color(r=r, g=g, b=b)


def color_distance(c1: sv.Color|None, c2: sv.Color|None) -> float:
    if c1 is None or c2 is None:
        return float('inf')
    v1 = np.array([c1.r, c1.g, c1.b])
    v2 = np.array([c2.r, c2.g, c2.b])
    return np.linalg.norm(v1-v2)


def resolve_goalkeepers_team_id(frame: np.ndarray, goalkeepers: sv.Detections,
                                team_a_color: sv.Color|None,
                                team_b_color: sv.Color|None,
                                threshold: float = COLOR_SIMILARITY_THRESHOLD) -> np.ndarray:
    ids = []
    fw = frame.shape[1]
    valid_colors = team_a_color and team_b_color
    for i in range(len(goalkeepers)):
        det = goalkeepers[i:i+1]
        cx = det.get_anchors_coordinates(sv.Position.CENTER)[0][0]
        assigned = -1
        if valid_colors:
            gc = calculate_average_color(frame, det)
            if gc:
                da = color_distance(gc, team_a_color)
                db = color_distance(gc, team_b_color)
                if abs(da-db) > threshold:
                    assigned = TEAM_A_ID if da<db else TEAM_B_ID
        if assigned==-1:
            assigned = TEAM_A_ID if cx<fw/2 else TEAM_B_ID
        ids.append(assigned)
    return np.array(ids, dtype=int)


def perform_ocr_on_crop(crop: np.ndarray) -> tuple[str|None, float|None]:
    if not PADDLEOCR_AVAILABLE or ocr_model is None or crop.size==0:
        return None, None
    try:
        res = ocr_model.ocr(crop, cls=False)
        best, best_conf = None, 0.0
        for item in (res[0] if res else []):
            text, conf = item[1]
            if text.isdigit() and MIN_JERSEY_DIGITS<=len(text)<=MAX_JERSEY_DIGITS and conf>OCR_CONFIDENCE_THRESHOLD:
                if conf>best_conf:
                    best, best_conf = text, conf
        return best, best_conf if best else None
    except Exception:
        return None, None


def process_frame(frame: np.ndarray, frame_idx: int):
    global player_data, recently_lost_jerseys, ball_positions

    # 1. Detection
    results = PLAYER_DETECTION_MODEL.predict(frame, conf=0.3, iou=0.5, device=DEVICE, verbose=False)
    if not results or len(results) == 0: return frame
    detections = sv.Detections.from_ultralytics(results[0])

    # 2. Pre-processing & Ball Position Update
    ball_detections = detections[detections.class_id == BALL_ID]
    people_detections = detections[detections.class_id != BALL_ID]

    # --- Update ball trail deque with outlier filtering ---
    if len(ball_detections) > 0 and ball_positions is not None:
        # Assuming single ball, take the first detection
        x1, y1, x2, y2 = ball_detections.xyxy[0]
        current_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        is_valid_position = True # Assume valid initially
        if len(ball_positions) > 0:
            # Compare with the last known valid position
            prev_center = ball_positions[-1]
            # Check if prev_center is a valid tuple before calculating distance
            if isinstance(prev_center, tuple) and len(prev_center) == 2:
                distance = math.dist(current_center, prev_center)
                # Check if distance exceeds threshold
                if distance > MAX_BALL_DISTANCE_PER_FRAME:
                    is_valid_position = False
                    # Optional: Log the outlier detection
                    # print(f"[Frame {frame_idx}] Ball position outlier detected. Dist: {distance:.1f} > {MAX_BALL_DISTANCE_PER_FRAME}. Skipping update.")
            # else: Invalid format in deque? Treat current as valid start.

        # Append only if the position is considered valid
        if is_valid_position:
            ball_positions.append(current_center)
        # If not valid, ball_positions is simply not updated for this frame

    # 3. Team/Role Classification
    players_detections = people_detections[people_detections.class_id == PLAYER_ID]
    goalkeepers_detections = people_detections[people_detections.class_id == GOALKEEPER_ID]
    referees_detections = people_detections[people_detections.class_id == REFEREE_ID]

    # --- Player Classification ---
    classified_players = sv.Detections.empty()
    if len(players_detections) > 0:
        players_crops = []; valid_indices = []
        for i, xyxy in enumerate(players_detections.xyxy):
            crop = sv.crop_image(frame, xyxy)
            if crop is not None and crop.size > 0: players_crops.append(crop); valid_indices.append(i)
        if players_crops:
             predicted_team_ids = team_classifier.predict(players_crops)
             if predicted_team_ids is not None and len(predicted_team_ids) == len(players_crops):
                 assigned_ids = np.full(len(players_detections), -1, dtype=int)
                 for i, pred_id in enumerate(predicted_team_ids): assigned_ids[valid_indices[i]] = pred_id
                 valid_classification_mask = (assigned_ids != -1)
                 players_detections.class_id = assigned_ids
                 classified_players = players_detections[valid_classification_mask]

    # --- Calculate Dynamic Team Colors (needed before GK resolution) ---
    team_a_detections = classified_players[classified_players.class_id == TEAM_A_ID]
    team_b_detections = classified_players[classified_players.class_id == TEAM_B_ID]
    current_team_a_color = calculate_average_color(frame, team_a_detections) or DEFAULT_TEAM_A_COLOR
    current_team_b_color = calculate_average_color(frame, team_b_detections) or DEFAULT_TEAM_B_COLOR
    current_referee_color = DEFAULT_REFEREE_COLOR
    dynamic_color_map = { TEAM_A_ID: current_team_a_color, TEAM_B_ID: current_team_b_color, REFEREE_TEAM_ID: current_referee_color }

    # --- Goalkeeper Classification (using enhanced function) ---
    classified_gks = sv.Detections.empty()
    if len(goalkeepers_detections) > 0:
        # *** UPDATED CALL ***
        gk_team_ids = resolve_goalkeepers_team_id(
            frame,
            goalkeepers_detections,
            current_team_a_color, # Pass calculated color
            current_team_b_color  # Pass calculated color
        )
        if gk_team_ids is not None and len(gk_team_ids) == len(goalkeepers_detections):
            valid_gk_mask = (gk_team_ids != -1) # Filter out GKs where resolution failed
            goalkeepers_detections.class_id = gk_team_ids
            classified_gks = goalkeepers_detections[valid_gk_mask] # Only keep successfully classified GKs
        else:
            print(f"[Frame {frame_idx}] Warning: Goalkeeper resolution returned unexpected result.")


    # --- Referee Classification ---
    classified_refs = sv.Detections.empty()
    if len(referees_detections) > 0:
        ref_team_ids = np.full(len(referees_detections), REFEREE_TEAM_ID)
        referees_detections.class_id = ref_team_ids
        classified_refs = referees_detections

    # --- Merge Detections for Tracking ---
    detections_to_track = sv.Detections.merge([classified_players, classified_gks, classified_refs])

    # 4. Tracking using BoTSORT (Same logic)
    tracked_detections = sv.Detections.empty()
    current_frame_tracker_ids = set()
    if len(detections_to_track) > 0 and tracker is not None:
        boxmot_input = np.hstack((detections_to_track.xyxy, detections_to_track.confidence[:, np.newaxis], detections_to_track.class_id[:, np.newaxis]))
        try:
            tracks = tracker.update(boxmot_input, frame)
            if tracks.shape[0] > 0:
                tracked_detections = sv.Detections(xyxy=tracks[:, 0:4], tracker_id=tracks[:, 4].astype(int), confidence=tracks[:, 5], class_id=tracks[:, 6].astype(int))
                current_frame_tracker_ids = set(tracked_detections.tracker_id)
        except Exception as e: print(f"[Frame {frame_idx}] Error during tracker update: {e}"); tracked_detections = sv.Detections.empty()
    elif tracker is not None:
         try: tracker.update(np.empty((0, 6)), frame)
         except Exception as e: print(f"[Frame {frame_idx}] Error updating tracker with empty input: {e}")

    # 5. OCR and Player ID Management (Label Generation)
    final_labels = []
    current_player_data = {}
    if len(tracked_detections) > 0:
        for i in range(len(tracked_detections)):
            track_id = tracked_detections.tracker_id[i]; team_id = tracked_detections.class_id[i]; bbox = tracked_detections.xyxy[i]
            x1, y1, x2, y2 = map(int, bbox); x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(width, x2), min(height, y2)
            detected_jersey_num, ocr_confidence = None, None; player_crop, gray_crop = None, None
            if x1 < x2 and y1 < y2:
                player_crop = frame[y1:y2, x1:x2]; gray_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
                detected_jersey_num, ocr_confidence = perform_ocr_on_crop(gray_crop)
                if detected_jersey_num is not None and player_crop is not None and gray_crop is not None:
                    try:
                        player_filename = os.path.join(OCR_DEBUG_DIR, f"frame{frame_idx}_track{track_id}_player.png")
                        ocr_input_filename = os.path.join(OCR_DEBUG_DIR, f"frame{frame_idx}_track{track_id}_ocr_input.png")
                        cv2.imwrite(player_filename, player_crop); cv2.imwrite(ocr_input_filename, gray_crop)
                    except Exception as write_e: print(f"[Frame {frame_idx}] Error saving OCR debug crop for track {track_id}: {write_e}")
            assigned_jersey_id = None
            if track_id in player_data:
                p_data = player_data[track_id]; p_data["last_seen"] = frame_idx; p_data["team_id"] = team_id
                current_jersey_id = p_data["jersey_id"]; mismatch_history = p_data["mismatch_history"]
                if detected_jersey_num is not None:
                    if current_jersey_id is None or detected_jersey_num == current_jersey_id: p_data["jersey_id"] = detected_jersey_num; p_data["jersey_confidence"] = ocr_confidence; mismatch_history.clear()
                    else:
                        mismatch_history.append(detected_jersey_num)
                        if len(mismatch_history) >= MISMATCH_CONSISTENCY_FRAMES and all(num == detected_jersey_num for num in mismatch_history): p_data["jersey_id"] = detected_jersey_num; p_data["jersey_confidence"] = ocr_confidence; mismatch_history.clear()
                else: mismatch_history.clear()
                assigned_jersey_id = p_data["jersey_id"]; current_player_data[track_id] = p_data
            else:
                found_match = False
                if detected_jersey_num is not None and detected_jersey_num in recently_lost_jerseys:
                    potential_matches = []
                    for lost_track_info in reversed(recently_lost_jerseys[detected_jersey_num]):
                        time_diff = frame_idx - lost_track_info["last_seen"]
                        if time_diff < LOST_TRACK_MEMORY_FRAMES and lost_track_info["team_id"] == team_id: potential_matches.append((lost_track_info, time_diff))
                    if potential_matches:
                        potential_matches.sort(key=lambda x: x[1]); best_match_info, _ = potential_matches[0]; assigned_jersey_id = detected_jersey_num
                        p_data = {"jersey_id": assigned_jersey_id, "jersey_confidence": ocr_confidence, "last_seen": frame_idx, "team_id": team_id, "mismatch_history": deque(maxlen=MISMATCH_CONSISTENCY_FRAMES)}
                        current_player_data[track_id] = p_data
                        try: recently_lost_jerseys[detected_jersey_num].remove(best_match_info)
                        except ValueError: pass
                        found_match = True
                if not found_match:
                    assigned_jersey_id = detected_jersey_num
                    current_player_data[track_id] = {"jersey_id": assigned_jersey_id, "jersey_confidence": ocr_confidence if detected_jersey_num is not None else None, "last_seen": frame_idx, "team_id": team_id, "mismatch_history": deque(maxlen=MISMATCH_CONSISTENCY_FRAMES)}
            if team_id == TEAM_A_ID: team_prefix = "T1"
            elif team_id == TEAM_B_ID: team_prefix = "T2"
            elif team_id == REFEREE_TEAM_ID: team_prefix = "Ref"
            else: team_prefix = f"T{team_id}"
            base_label = f"{team_prefix} P{track_id}"; display_id = base_label
            if assigned_jersey_id is not None: display_id = f"{base_label} #{assigned_jersey_id}"
            final_labels.append(display_id)

    # 6. Update Global Player Data & Handle Lost Tracks (Same as before)
    lost_tracker_ids = set(player_data.keys()) - current_frame_tracker_ids
    for lost_id in lost_tracker_ids:
        lost_info = player_data[lost_id]
        if lost_info.get("jersey_id") is not None: recently_lost_jerseys[lost_info["jersey_id"]].append({"tracker_id": lost_id, "last_seen": lost_info["last_seen"], "team_id": lost_info["team_id"]})
    if frame_idx > 0 and fps > 0 and frame_idx % (int(fps) * 60) == 0:
        for jersey_num in list(recently_lost_jerseys.keys()):
            q = recently_lost_jerseys[jersey_num]
            valid_entries = deque([entry for entry in q if (frame_idx - entry["last_seen"]) < LOST_TRACK_MEMORY_FRAMES * 2], maxlen=10)
            if valid_entries: recently_lost_jerseys[jersey_num] = valid_entries
            else: del recently_lost_jerseys[jersey_num]
    player_data = current_player_data

    # 7. Annotation
    annotated_frame = frame.copy()

    # --- Annotate "Magical" Ball Trail ---
    if ball_positions is not None and len(ball_positions) >= 2:
        num_points = len(ball_positions)
        for i in range(1, num_points):
            pt1 = ball_positions[i-1]; pt2 = ball_positions[i]
            if isinstance(pt1, tuple) and isinstance(pt2, tuple) and len(pt1) == 2 and len(pt2) == 2:
                 cv2.line(annotated_frame, pt1, pt2, BALL_TRAIL_BASE_COLOR, BALL_TRAIL_THICKNESS)
                 alpha_fraction = (i - 1) / max(1, num_points - 1)
                 sparkle_intensity = int(SPARKLE_BASE_INTENSITY + (SPARKLE_MAX_INTENSITY - SPARKLE_BASE_INTENSITY) * alpha_fraction)
                 sparkle_color = (sparkle_intensity, sparkle_intensity, sparkle_intensity)
                 for _ in range(SPARKLE_COUNT):
                     offset_x = random.randint(-SPARKLE_OFFSET, SPARKLE_OFFSET); offset_y = random.randint(-SPARKLE_OFFSET, SPARKLE_OFFSET)
                     sparkle_pt = (pt2[0] + offset_x, pt2[1] + offset_y)
                     cv2.circle(annotated_frame, sparkle_pt, SPARKLE_RADIUS, sparkle_color, -1)

    # --- Annotate Current Ball Position ---
    if ball_positions is not None and len(ball_positions) > 0:
         last_pos = ball_positions[-1]
         if isinstance(last_pos, tuple) and len(last_pos) == 2:
              cv2.circle(annotated_frame, last_pos, CURRENT_BALL_MARKER_RADIUS, CURRENT_BALL_MARKER_COLOR, CURRENT_BALL_MARKER_THICKNESS)

    # --- Annotate Tracked People ---
    if len(tracked_detections) > 0:
        if len(final_labels) == len(tracked_detections):
            unique_team_ids = np.unique(tracked_detections.class_id)
            for current_team_id in unique_team_ids:
                team_mask = (tracked_detections.class_id == current_team_id); team_detections = tracked_detections[team_mask]
                team_labels = [label for i, label in enumerate(final_labels) if team_mask[i]]
                if len(team_detections) == 0: continue
                team_color = dynamic_color_map.get(current_team_id, FALLBACK_COLOR)
                temp_ellipse_annotator = sv.EllipseAnnotator(color=team_color, thickness=ELLIPSE_THICKNESS)
                temp_label_annotator = sv.LabelAnnotator(color=team_color, text_color=LABEL_TEXT_COLOR, text_position=LABEL_TEXT_POSITION, text_scale=LABEL_TEXT_SCALE, text_thickness=LABEL_TEXT_THICKNESS)
                try:
                    annotated_frame = temp_ellipse_annotator.annotate(annotated_frame, team_detections)
                    annotated_frame = temp_label_annotator.annotate(annotated_frame, team_detections, team_labels)
                except Exception as e: print(f"[Frame {frame_idx}] Error during annotation for team {current_team_id} (Color: {team_color.as_hex()}): {e}")

    return annotated_frame



def process_video(input_path: str, output_path: str):
    # Setup video info
    try:
        info = sv.VideoInfo.from_video_path(input_path)
        w, h, fps = info.width, info.height, info.fps
        total = info.total_frames or 0
    except:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    # Tracking and OCR init
    print(f"Initializing tracker and OCR...")
    tracker = BotSort(
        reid_weights=(REID_WEIGHTS_PATH) if REID_WEIGHTS_PATH.exists() else None,
        device=DEVICE, half=False, with_reid=REID_WEIGHTS_PATH.exists()
    )
    print(f"Tracker initialized.")
    team_classifier = __import__('sports.common.team', fromlist=['TeamClassifier']).TeamClassifier(device=DEVICE.type)

    # Prepare video writer and frame generator
    gen = sv.get_video_frames_generator(source_path=input_path, stride=1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    player_data = {}
    recently_lost = defaultdict(lambda: deque(maxlen=10))
    # Ball trail
    trail_len = int(fps)
    ball_positions = deque(maxlen=trail_len)

    with tqdm(total=total, desc="Processing video", unit="frame") as pbar:
        for idx, frame in enumerate(gen):
            if frame is None:
                break
            try:
                annotated, player_data = process_frame(
                    frame, idx, tracker, team_classifier,
                    ball_positions, w, h, fps,
                    player_data, recently_lost
                )
                writer.write(annotated)
            except KeyboardInterrupt:
                print("Interrupted by user.")
                break
            except Exception as e:
                print(f"Error on frame {idx}: {e}")
                traceback.print_exc()
                writer.write(frame)
            pbar.update(1)

    writer.release()
    print(f"Finished. Output saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process and annotate football videos.")
    parser.add_argument("--input", type=str, required=True, help="Input video file path")
    parser.add_argument("--output", type=str, required=True, help="Destination output video file path")
    args = parser.parse_args()
    process_video(args.input, args.output)
