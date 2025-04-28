import numpy as np
import supervision as sv
import torch
from transformers import AutoProcessor, SiglipVisionModel
from sklearn.cluster import KMeans
import umap # Use umap-learn
from more_itertools import chunked
from tqdm import tqdm
import logging

from app.core import config
from app.processing.utils import calculate_average_color, color_distance
from app.processing.detection import ObjectDetector # Needed for initial crop collection

# Configure logging
logger = logging.getLogger(__name__)

class TeamClassifier:
    """
    Handles training a classifier based on player appearance embeddings
    and predicting team assignments for player detections.
    """
    def __init__(self, device=config.DEVICE):
        self.device = device
        self.embeddings_model = None
        self.embeddings_processor = None
        self.reducer = None # UMAP reducer
        self.clustering_model = None # KMeans model
        self._load_embeddings_model()
        self.is_trained = False

    def _load_embeddings_model(self):
        """Loads the Siglip vision model and processor."""
        try:
            self.embeddings_model = SiglipVisionModel.from_pretrained(
                config.SIGLIP_MODEL_PATH
            ).to(self.device)
            self.embeddings_processor = AutoProcessor.from_pretrained(
                config.SIGLIP_MODEL_PATH
            )
            logger.info(f"Siglip model loaded successfully to {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load Siglip model: {e}", exc_info=True)
            # Allow continuation without embedding-based classification if model fails
            self.embeddings_model = None
            self.embeddings_processor = None

    def _extract_embeddings(self, crops: list) -> np.ndarray | None:
        """Extracts embeddings for a list of image crops."""
        if not self.embeddings_model or not self.embeddings_processor:
            logger.warning("Embeddings model not available. Cannot extract embeddings.")
            return None
        if not crops:
            return np.empty((0, self.embeddings_model.config.hidden_size)) # Return empty array of correct shape

        # Convert cv2 images (numpy arrays) to PIL Images if needed
        pil_crops = [sv.cv2_to_pillow(crop) for crop in crops if crop is not None and crop.size > 0]
        if not pil_crops:
            logger.warning("No valid crops provided for embedding extraction.")
            return np.empty((0, self.embeddings_model.config.hidden_size))

        all_embeddings = []
        batches = chunked(pil_crops, config.TEAM_CLASSIFIER_BATCH_SIZE)

        with torch.no_grad():
            for batch in tqdm(batches, desc="Embedding Extraction", leave=False):
                try:
                    inputs = self.embeddings_processor(
                        images=list(batch), return_tensors="pt", padding=True
                    ).to(self.device)
                    outputs = self.embeddings_model(**inputs)
                    # Use pooler_output if available, otherwise mean of last hidden state
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                         embeddings = outputs.pooler_output.cpu().numpy()
                    else:
                         # Mean pooling of the last hidden state
                         embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    all_embeddings.append(embeddings)
                except Exception as e:
                    logger.error(f"Error processing batch for embeddings: {e}", exc_info=True)
                    # Add placeholder embeddings (e.g., zeros) or skip batch
                    # Adding zeros might skew results, skipping might be safer if errors are rare
                    # For simplicity here, we'll skip, but a robust solution might need placeholders
                    continue # Skip batch on error

        if not all_embeddings:
            logger.warning("No embeddings could be extracted.")
            return np.empty((0, self.embeddings_model.config.hidden_size))

        return np.concatenate(all_embeddings)

    def train(self, video_path: str, detector: ObjectDetector):
        """
        Trains the team classifier by collecting player crops, extracting embeddings,
        and fitting clustering models (UMAP + KMeans).

        Args:
            video_path: Path to the video file.
            detector: An initialized ObjectDetector instance.
        """
        if not self.embeddings_model:
            logger.warning("Embeddings model not loaded. Skipping training.")
            self.is_trained = False
            return

        logger.info("Starting team classifier training...")
        frame_generator = sv.get_video_frames_generator(
            source_path=video_path, stride=config.TEAM_CLASSIFIER_STRIDE
        )

        all_player_crops = []
        # Use total frames from video info for progress bar if available
        video_info = sv.VideoInfo.from_video_path(video_path)
        total_initial_frames = (video_info.total_frames // config.TEAM_CLASSIFIER_STRIDE
                                if video_info.total_frames else None)

        logger.info(f"Collecting initial player crops (stride={config.TEAM_CLASSIFIER_STRIDE})...")
        for frame in tqdm(frame_generator, total=total_initial_frames, desc="Collecting Crops"):
            if frame is None: continue
            detections = detector.predict(frame)
            player_detections = detections[detections.class_id == config.PLAYER_ID]
            # Apply NMS specifically to players before cropping
            player_detections = player_detections.with_nms(threshold=config.NMS_THRESHOLD)

            for xyxy in player_detections.xyxy:
                crop = sv.crop_image(image=frame, xyxy=xyxy)
                if crop is not None and crop.size > 0:
                    all_player_crops.append(crop)

        if not all_player_crops:
            logger.warning("No player crops collected. Cannot train team classifier.")
            self.is_trained = False
            return

        logger.info(f"Collected {len(all_player_crops)} player crops. Extracting embeddings...")
        embeddings = self._extract_embeddings(all_player_crops)

        if embeddings is None or embeddings.shape[0] < 2: # Need at least 2 samples for clustering
             logger.warning(f"Insufficient embeddings ({embeddings.shape[0] if embeddings is not None else 0}) "
                            f"extracted. Cannot train clustering model.")
             self.is_trained = False
             return

        logger.info(f"Extracted {embeddings.shape[0]} embeddings. Training UMAP and KMeans...")
        try:
            # Reduce dimensionality with UMAP (optional but helpful)
            # Adjust n_neighbors and min_dist based on dataset size/density
            n_neighbors = min(15, embeddings.shape[0] - 1) # Ensure n_neighbors < n_samples
            if n_neighbors < 2:
                 logger.warning(f"Too few samples ({embeddings.shape[0]}) for UMAP. Skipping dimensionality reduction.")
                 projections = embeddings # Use raw embeddings
            else:
                self.reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
                projections = self.reducer.fit_transform(embeddings)

            # Cluster the projections (or raw embeddings)
            self.clustering_model = KMeans(n_clusters=2, random_state=42, n_init=10) # n_init='auto' in newer sklearn
            self.clustering_model.fit(projections)
            self.is_trained = True
            logger.info("Team classifier training complete (UMAP + KMeans).")

        except Exception as e:
            logger.error(f"Error during UMAP/KMeans training: {e}", exc_info=True)
            self.is_trained = False
            self.reducer = None
            self.clustering_model = None


    def predict(self, player_crops: list) -> np.ndarray | None:
        """
        Predicts team IDs (0 or 1 relative to training clusters) for new player crops.

        Args:
            player_crops: A list of player crop images (NumPy arrays).

        Returns:
            A NumPy array of predicted cluster IDs (0 or 1), or None if not trained
            or prediction fails. Returns array of -1s if embeddings fail.
        """
        if not self.is_trained or not self.clustering_model:
            # logger.warning("Classifier not trained. Cannot predict.")
             # Fallback: Return -1 for all crops, indicating unknown team
            return np.full(len(player_crops), -1, dtype=int)

        if not player_crops:
            return np.array([], dtype=int)

        embeddings = self._extract_embeddings(player_crops)

        if embeddings is None or embeddings.shape[0] == 0:
             logger.warning("Could not extract embeddings for prediction.")
             return np.full(len(player_crops), -1, dtype=int) # Indicate failure

        try:
            # Apply UMAP transformation if reducer was trained
            if self.reducer:
                projections = self.reducer.transform(embeddings)
            else:
                projections = embeddings # Use raw embeddings if UMAP failed/skipped

            # Predict clusters using the trained KMeans model
            cluster_ids = self.clustering_model.predict(projections)

            # Map cluster IDs (0, 1) directly to Team A/B IDs (0, 1)
            # This assumes cluster 0 corresponds to Team A, cluster 1 to Team B.
            # A more robust system might involve checking average colors of clusters
            # against expected jersey colors after training, but this is simpler.
            predicted_team_ids = np.where(cluster_ids == 0, config.TEAM_A_ID, config.TEAM_B_ID)
            return predicted_team_ids

        except Exception as e:
            logger.error(f"Error during team prediction: {e}", exc_info=True)
            return np.full(len(player_crops), -1, dtype=int) # Indicate failure


def resolve_goalkeepers_team_id(
    frame: np.ndarray,
    goalkeepers: sv.Detections,
    team_a_color: sv.Color | None,
    team_b_color: sv.Color | None,
) -> np.ndarray:
    """
    Assigns team IDs to goalkeepers based primarily on color similarity,
    with a positional fallback for ambiguous cases.

    Args:
        frame: The current video frame.
        goalkeepers: sv.Detections object containing only goalkeeper detections.
        team_a_color: The calculated average color for Team A players in the frame.
        team_b_color: The calculated average color for Team B players in the frame.

    Returns:
        A NumPy array of team IDs (config.TEAM_A_ID or config.TEAM_B_ID)
        corresponding to each goalkeeper detection. Returns empty array if no GKs.
        Uses -1 for GKs where assignment fails completely.
    """
    if len(goalkeepers) == 0:
        return np.array([], dtype=int)

    goalkeeper_team_ids = np.full(len(goalkeepers), -1, dtype=int) # Initialize with -1
    frame_height, frame_width, _ = frame.shape
    valid_team_colors = team_a_color is not None and team_b_color is not None

    for i in range(len(goalkeepers)):
        gk_detection_single = goalkeepers[i:i+1] # Process one GK at a time
        gk_center_x, _ = gk_detection_single.get_anchors_coordinates(sv.Position.CENTER)[0]
        assigned_id = -1 # Default to invalid ID for this GK

        # 1. Try Color Similarity if possible
        if valid_team_colors:
            gk_color = calculate_average_color(
                frame, gk_detection_single, config.CENTRAL_FRACTION_FOR_COLOR
            )

            if gk_color is not None:
                dist_a = color_distance(gk_color, team_a_color)
                dist_b = color_distance(gk_color, team_b_color)

                # Check if colors are distinct enough
                if abs(dist_a - dist_b) > config.COLOR_SIMILARITY_THRESHOLD:
                    assigned_id = config.TEAM_A_ID if dist_a < dist_b else config.TEAM_B_ID
                # else: Colors are ambiguous, fall back to position

            # else: gk_color calculation failed, fall back to position

        # 2. Positional Fallback (if color failed or was ambiguous)
        if assigned_id == -1:
            # logger.debug(f"GK {i} using positional fallback.") # Optional debug
            # Assign based on which half of the pitch they are on
            # Assumes Team A (ID 0) defends left goal, Team B (ID 1) defends right
            assigned_id = config.TEAM_A_ID if gk_center_x < frame_width / 2 else config.TEAM_B_ID

        goalkeeper_team_ids[i] = assigned_id

    return goalkeeper_team_ids


def classify_detections(
    frame: np.ndarray,
    detections: sv.Detections,
    team_classifier: TeamClassifier | None
) -> tuple[sv.Detections, dict[int, sv.Color]]:
    """
    Classifies player, goalkeeper, and referee detections.

    Args:
        frame: The current video frame.
        detections: All detections from the ObjectDetector for the frame.
        team_classifier: An initialized (and preferably trained) TeamClassifier instance.

    Returns:
        A tuple containing:
        - sv.Detections: Detections object with class_id updated to team/role IDs
                         (TEAM_A_ID, TEAM_B_ID, REFEREE_TEAM_ID). Only includes
                         detections that could be classified.
        - dict[int, sv.Color]: A dictionary mapping team/role IDs to their
                                calculated or default colors for this frame.
    """
    # Separate detections by initial class ID
    players = detections[detections.class_id == config.PLAYER_ID]
    goalkeepers = detections[detections.class_id == config.GOALKEEPER_ID]
    referees = detections[detections.class_id == config.REFEREE_ID]

    classified_players = sv.Detections.empty()
    predicted_team_ids = None

    # --- Player Classification ---
    if len(players) > 0 and team_classifier and team_classifier.is_trained:
        player_crops = []
        valid_indices = [] # Keep track of which original player detections are valid
        for i, xyxy in enumerate(players.xyxy):
            crop = sv.crop_image(frame, xyxy)
            if crop is not None and crop.size > 0:
                player_crops.append(crop)
                valid_indices.append(i)
            # else: logger.debug(f"Skipping invalid crop for player detection {i}")

        if player_crops:
            predicted_team_ids = team_classifier.predict(player_crops)

            if predicted_team_ids is not None and len(predicted_team_ids) == len(valid_indices):
                 # Create a full array of -1s for all original players
                 assigned_ids = np.full(len(players), -1, dtype=int)
                 # Fill in the predicted IDs at the valid indices
                 assigned_ids[valid_indices] = predicted_team_ids

                 # Filter players where prediction was successful (not -1)
                 valid_classification_mask = (assigned_ids != -1)
                 players.class_id = assigned_ids # Assign IDs (including -1s)
                 classified_players = players[valid_classification_mask] # Keep only successfully classified
                 # logger.debug(f"Classified {len(classified_players)} players.")
            else:
                 logger.warning("Team classifier prediction failed or returned incorrect shape.")
                 # Keep players with original class ID if classification fails? Or discard?
                 # For now, we treat them as unclassified, so classified_players remains empty.
    elif len(players) > 0:
         logger.warning("Team classifier not available or not trained. Players remain unclassified by team.")
         # classified_players remains empty

    # --- Calculate Dynamic Team Colors (Based on classified players) ---
    team_a_detections = classified_players[classified_players.class_id == config.TEAM_A_ID]
    team_b_detections = classified_players[classified_players.class_id == config.TEAM_B_ID]

    current_team_a_color = calculate_average_color(frame, team_a_detections, config.CENTRAL_FRACTION_FOR_COLOR) or config.DEFAULT_TEAM_A_COLOR
    current_team_b_color = calculate_average_color(frame, team_b_detections, config.CENTRAL_FRACTION_FOR_COLOR) or config.DEFAULT_TEAM_B_COLOR
    current_referee_color = config.DEFAULT_REFEREE_COLOR # Referees usually have a consistent color

    dynamic_color_map = {
        config.TEAM_A_ID: current_team_a_color,
        config.TEAM_B_ID: current_team_b_color,
        config.REFEREE_TEAM_ID: current_referee_color
    }

    # --- Goalkeeper Classification ---
    classified_gks = sv.Detections.empty()
    if len(goalkeepers) > 0:
        gk_team_ids = resolve_goalkeepers_team_id(
            frame,
            goalkeepers,
            current_team_a_color, # Use dynamically calculated color
            current_team_b_color  # Use dynamically calculated color
        )
        if gk_team_ids is not None and len(gk_team_ids) == len(goalkeepers):
            valid_gk_mask = (gk_team_ids != -1) # Filter out GKs where resolution failed
            goalkeepers.class_id = gk_team_ids # Assign resolved IDs
            classified_gks = goalkeepers[valid_gk_mask] # Keep only successfully classified GKs
            # logger.debug(f"Classified {len(classified_gks)} goalkeepers.")
        else:
            logger.warning("Goalkeeper resolution returned unexpected result.")
            # classified_gks remains empty

    # --- Referee Classification ---
    classified_refs = sv.Detections.empty()
    if len(referees) > 0:
        # Assign the dedicated referee team ID
        ref_team_ids = np.full(len(referees), config.REFEREE_TEAM_ID)
        referees.class_id = ref_team_ids
        classified_refs = referees # Assume all referee detections are valid
        # logger.debug(f"Classified {len(classified_refs)} referees.")


    # --- Merge Classified Detections ---
    # Combine players, goalkeepers, and referees that have been successfully assigned a team/role ID
    final_classified_detections = sv.Detections.merge([
        classified_players,
        classified_gks,
        classified_refs
    ])
    # logger.debug(f"Total classified detections for tracking: {len(final_classified_detections)}")

    return final_classified_detections, dynamic_color_map

