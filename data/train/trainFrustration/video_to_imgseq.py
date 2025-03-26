import cv2
import os

def video_to_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video and save them as images.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Directory where frames will be saved.
        frame_rate (int): Extract 1 frame every 'frame_rate' seconds.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    frame_interval = fps * frame_rate  

    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} into {output_folder}")

def process_all_videos(folder_path, frame_rate=0.1):
    """
    Process all video files in a folder.

    Args:
        folder_path (str): Path to the folder containing videos.
        frame_rate (int): Extract 1 frame every 'frame_rate' seconds.

    Returns:
        None
    """
    video_extensions = {".mp4"} # add more extensions of u want
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(folder_path, filename)
            output_folder = os.path.join(folder_path, f"frames_{os.path.splitext(filename)[0]}")
            video_to_frames(video_path, output_folder, frame_rate)

video_folder = os.path.dirname(os.path.abspath(__file__))
process_all_videos(video_folder, frame_rate=1)
