import os
import librosa
import random
import kagglehub

import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Download latest version
path = kagglehub.dataset_download("kinguistics/heartbeat-sounds")

print("Path to dataset files:", path)

def save_mel_spectrogram(wav_path, output_path):
    waveform, sr = librosa.load(wav_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Chuẩn hóa về [0, 1]
    norm_mel = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    # Chuyển colormap sang ảnh RGB (0-255)
    colormap = plt.colormaps['plasma']  
    colored = colormap(norm_mel)
    rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)  # Bỏ kênh alpha

    # Lưu ảnh RGB
    Image.fromarray(rgb_image).save(output_path)

# Prepare pipeline
original_path = "/Users/vominhthinh/.cache/kagglehub/datasets/kinguistics/heartbeat-sounds/versions/1/set_a"
output_base = "heartbeat_dataset"  # chứa ảnh spectrogram

# Tên các class
classes = ["normal", "murmur", "extrahls", "artifact"]
splits = ["train", "test"]

# Tạo thư mục output
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_base, split, cls), exist_ok=True)

# Gom file theo class
class_to_files = {cls: [] for cls in classes}
for filename in os.listdir(original_path):
    if filename.endswith(".wav"):
        for cls in classes:
            if filename.startswith(cls):
                class_to_files[cls].append(filename)
                break

# Chia train/test và lưu ảnh spectrogram
split_ratio = 0.8

for cls, files in class_to_files.items():
    random.shuffle(files)
    split_idx = int(len(files) * split_ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    for split, file_list in zip(["train", "test"], [train_files, test_files]):
        for fname in file_list:
            wav_path = os.path.join(original_path, fname)
            output_path = os.path.join(output_base, split, cls, fname.replace(".wav", ".png"))
            save_mel_spectrogram(wav_path, output_path)

print("Done: data has splited into train/test and classified by class name.")