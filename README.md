# Football Player Tracking and Team Classification

This repository contains a modular pipeline for football player tracking and team classification. It uses YOLO for object detection, BotSort (from BoxMOT) for player tracking, and a team classifier based on SIGLIP embeddings.

## Features

- Detection of players, goalkeepers, referees, and ball
- Tracking players with unique IDs across frames
- Automatic team classification using visual embeddings
- Full video processing with annotated output

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mikel-brostrom/boxmot.git
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required model weights:
   - YOLO detection model: Should be located at `app/models/yolo11_football_v2/weights/best.pt`
   - ReID model for tracking: `osnet_x0_25_msmt17.pt` (included in the repository)

## Usage

Run the football tracking pipeline with the following command:

```bash
python football_tracker.py --source path/to/video.mp4 --output path/to/output.mp4
```

### Command-line Arguments

- `--source`, `-s`: Path to the source video (required)
- `--output`, `-o`: Path for the output video (optional, default: source_filename-result.mp4)
- `--model`, `-m`: Path to the YOLO detection model (default: app/models/yolo11_football_v2/weights/best.pt)
- `--reid`, `-r`: Path to the ReID weights for BotSort tracker (default: osnet_x0_25_msmt17.pt)
- `--device`, `-d`: Device to run inference on (default: cuda if available, else cpu)
- `--conf-threshold`: Confidence threshold for detections (default: 0.3)
- `--stride`: Frame stride for team classification training (default: 30)

## Project Structure

- `football_tracker.py`: Main script to run the football tracking pipeline
- `sports/common/detection.py`: Module for object detection and tracking
- `sports/common/team.py`: Module for team classification
- `sports/common/visualize.py`: Module for visualization and annotation

## Examples

Process a football match video and save the annotated output:

```bash
python football_tracker.py --source app/test_data/raw/121364_0.mp4
```

## Acknowledgements

This project uses the following libraries:
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) for multiple object tracking
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [Supervision](https://github.com/roboflow/supervision) for computer vision tools
- [Transformers](https://github.com/huggingface/transformers) for visual embeddings
