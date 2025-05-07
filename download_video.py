from utils import download_from_s3

if __name__ == "__main__":
    bucket_name = "football.academy.dataset.using.tactical.camera101"
    object_key = "ScreenRecording_05-04-2025 23-32-29_1.MP4"
    download_path = "ScreenRecording_05-04-2025_23-32-29_1.MP4"

    download_from_s3(bucket_name, object_key, download_path)