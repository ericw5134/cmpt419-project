import os
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def select_key_frames_from_folder(frame_dir, threshold=0.3, max_frames=16):
    """
    Select key frames from a folder using SSIM.

    Args:
        frame_dir: Directory containing .jpg frame images
        threshold: SSIM threshold for visual difference
        max_frames: Maximum number of frames to return

    Returns:
        List of selected frame paths
    """
    frame_files = sorted([
        os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")
    ])

    if not frame_files:
        return []

    # Convert first frame
    prev_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
    selected_paths = [frame_files[0]]
    count = 1

    for path in frame_files[1:]:
        curr_frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if curr_frame is None or prev_frame is None:
            continue

        score = ssim(prev_frame, curr_frame)
        if score < threshold:
            selected_paths.append(path)
            prev_frame = curr_frame
            count += 1

        if count >= max_frames:
            break

    return selected_paths


def sample_evenly_spaced_frames(frame_dir, num_frames=16):
    frame_files = sorted([
        os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")
    ])

    if not frame_files:
        return []

    total = len(frame_files)
    if total <= num_frames:
        # Repeat last frame if too few
        frame_files += [frame_files[-1]] * (num_frames - total)
        return frame_files

    # Sample evenly spaced frames
    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    return [frame_files[i] for i in indices]