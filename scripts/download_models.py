#!/usr/bin/env python3
"""
Script to download model files that are not included in the Git repository.
These files are stored externally due to their large size.
"""

import os
import argparse
import requests
from pathlib import Path
import hashlib

# Define model information
MODELS = {
    "yolov11_football_best.pt": {
        "url": "REPLACE_WITH_YOUR_DOWNLOAD_LINK",  # Replace with your actual download link (Google Drive, S3, etc.)
        "target_path": "app/models/",
        "md5": "REPLACE_WITH_MD5_HASH",  # Optional: for verification
    }
}

def download_file(url, destination):
    """Download a file from a URL to a destination path."""
    print(f"Downloading {url} to {destination}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download with progress
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    
    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                done = int(50 * downloaded / total_size) if total_size > 0 else 0
                print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded}/{total_size} bytes", 
                      end='', flush=True)
    
    print("\nDownload complete!")

def verify_md5(file_path, expected_md5):
    """Verify the MD5 hash of a file."""
    if not expected_md5:
        print("No MD5 hash provided for verification, skipping...")
        return True
        
    print(f"Verifying MD5 hash of {file_path}...")
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    file_md5 = md5_hash.hexdigest()
    if file_md5 == expected_md5:
        print("MD5 verification successful!")
        return True
    else:
        print(f"MD5 verification failed! Expected {expected_md5}, got {file_md5}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download model files for the football analysis project.')
    parser.add_argument('--model', choices=list(MODELS.keys()) + ['all'], default='all',
                      help='Specify which model to download (default: all)')
    args = parser.parse_args()
    
    # Get the repository root directory
    repo_root = Path(__file__).parent.parent.absolute()
    
    models_to_download = MODELS.items() if args.model == 'all' else [(args.model, MODELS[args.model])]
    
    for model_name, model_info in models_to_download:
        target_path = os.path.join(repo_root, model_info['target_path'], model_name)
        
        # Check if the file already exists
        if os.path.exists(target_path):
            print(f"Model {model_name} already exists at {target_path}")
            # Optionally verify existing file
            if model_info.get('md5') and verify_md5(target_path, model_info['md5']):
                continue
            else:
                print("Redownloading...")
        
        # Download the file
        download_file(model_info['url'], target_path)
        
        # Verify the download
        if model_info.get('md5'):
            verify_md5(target_path, model_info['md5'])

if __name__ == "__main__":
    main()