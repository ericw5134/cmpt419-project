import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchaudio

from utility.frame_sampler import sample_evenly_spaced_frames

class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None, audio_transform=None):
        """
        Args:
            root_dir (str): Root path to either train or test folder
            transform (callable, optional): Optional transform to be applied to images
            audio_transform (callable, optional): Optional transform to be applied to audio
        """
        self.root_dir = root_dir
        self.transform = transform
        self.audio_transform = audio_transform
        self.data = []
        self.label_to_idx = {'Rage': 0, 'Excitement': 1, 'Fear': 2, 'Frustration': 3}

        emotions = os.listdir(os.path.join(root_dir, "Audio"))
        for folder in os.listdir(os.path.join(root_dir, "Audio")):
            audio_path = os.path.join(root_dir, "Audio", folder, f"{folder}.wav")
            name, emotion_number = folder.rsplit("_", 1)
            for emotion in ["Rage", "Fear", "Excitement", "Frustration"]:
                frame_folder_name = f"frames_{folder}"
                frame_dir = os.path.join(root_dir, emotion, frame_folder_name)
                if os.path.isdir(frame_dir):
                    self.data.append((audio_path, frame_dir, emotion))
                    break  # Found the matching emotion folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, frame_dir, label = self.data[idx]

        # === Load and process audio ===
        spectrogram = torch.zeros(1, 64, 64)  # default silent spectrogram

        try:
            abs_audio_path = os.path.abspath(os.path.normpath(audio_path))
            if os.path.exists(abs_audio_path):
                waveform, _ = torchaudio.load(abs_audio_path, backend="soundfile")

                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Standardize length
                target_length = 160000
                waveform = waveform[:, :target_length]
                if waveform.shape[1] < target_length:
                    pad_size = target_length - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad_size))

                # Convert to spectrogram
                if self.audio_transform:
                    spectrogram = self.audio_transform(waveform)  # [1, 64, time]

                    # Resize to [1, 64, 64] if needed
                    if spectrogram.shape[-1] < 64:
                        pad = 64 - spectrogram.shape[-1]
                        spectrogram = torch.nn.functional.pad(spectrogram, (0, pad))
                    elif spectrogram.shape[-1] > 64:
                        spectrogram = spectrogram[:, :, :64]
        except Exception as e:
            print(f"[Audio Load Error] {audio_path}: {e}")

        # === Load and process frames ===
        try:
            frame_paths = sample_evenly_spaced_frames(frame_dir, num_frames=20)
            frames = [Image.open(p).convert("L") for p in frame_paths]
            frames = [self.transform(f) for f in frames]
            frames_tensor = torch.stack(frames)  # [20, 1, 112, 112]
        except Exception as e:
            print(f"[Frame Load Error] {frame_dir}: {e}")
            blank_frame = self.transform(Image.new('L', (112, 112)))
            frames_tensor = torch.stack([blank_frame] * 20)

        return spectrogram, frames_tensor, self.label_to_idx[label]




