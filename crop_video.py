from moviepy.editor import VideoFileClip

def crop_video(input_path, output_path, start_time, end_time):
    """
    Crop a video to the specified start and end time.

    :param input_path: Path to the input video file
    :param output_path: Path to save the cropped video
    :param start_time: Start time in seconds
    :param end_time: End time in seconds
    """
    # Load video
    video = VideoFileClip(input_path)

    # Crop/Trim video
    cropped_video = video.subclip(start_time, end_time)

    # Save the result
    cropped_video.write_videofile(output_path, codec="libx264")

if __name__ == "__main__":
    input_video = "ScreenRecording_05-04-2025_23-32-29_1.MP4"
    output_video = "cropped_ScreenRecording_05-04-2025_23-32-29_1.MP4"
    start_time = 10  # Start time in seconds
    end_time = 300   # End time in seconds

    crop_video(input_video, output_video, start_time, end_time)