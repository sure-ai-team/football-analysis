#!/bin/bash
# Setup script for football tracking pipeline

# Create necessary directories
mkdir -p app/models/yolo11_football_v2/weights

# Check if OSNET model exists, download if not
if [ ! -f "osnet_x0_25_msmt17.pt" ]; then
    echo "Downloading OSNet ReID model for tracking..."
    wget https://github.com/mikel-brostrom/assets/releases/download/v1.0.0/osnet_x0_25_msmt17.pt
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install BoxMOT if not installed
if ! pip show boxmot > /dev/null; then
    echo "Installing BoxMOT from GitHub..."
    pip install -e git+https://github.com/mikel-brostrom/boxmot.git#egg=boxmot
fi

# Verify that the YOLO model exists
if [ ! -f "app/models/yolo11_football_v2/weights/best.pt" ]; then
    echo "YOLO model not found at app/models/yolo11_football_v2/weights/best.pt"
    echo "Please ensure you have the correct model or update the path in the football_tracker.py script"
fi

echo "Setup completed! You can now run the football tracking pipeline."
echo "Example: ./football_tracker.py --source app/test_data/raw/121364_0.mp4"