import os
from moviepy import *

def convert_mp4_to_wav(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            mp4_file_path = os.path.join(directory, filename)
            output_folder = os.path.splitext(mp4_file_path)[0] 
            os.makedirs(output_folder, exist_ok=True)  
            wav_file_name = os.path.splitext(filename)[0] + ".wav" 
            wav_file_path = os.path.join(output_folder, wav_file_name)  
            video_clip = VideoFileClip(mp4_file_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(wav_file_path)
            audio_clip.close()
            video_clip.close()
            print(f"Converted {mp4_file_path} to {wav_file_path}")

current_directory = os.getcwd()
convert_mp4_to_wav(current_directory)