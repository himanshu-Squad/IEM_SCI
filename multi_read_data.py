import numpy as np
import torch
import torch.utils.data
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import os
import cv2  # Add this import for video processing

batch_w = 600
batch_h = 400


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, video_dir, task):
        self.video_dir = video_dir
        self.task = task
        self.video_files = []

        for root, dirs, names in os.walk(self.video_dir):
            for name in names:
                if name.endswith(('.mp4', '.avi', '.mov')):  # Add video file extensions
                    self.video_files.append(os.path.join(root, name))

        self.video_files.sort()
        self.count = len(self.video_files)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_video_frames(self, file):
        cap = cv2.VideoCapture(file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = Image.fromarray(frame)
            frame = self.transform(frame).numpy()
            frames.append(frame)
        cap.release()
        return frames

    def __getitem__(self, index):
        frames = self.load_video_frames(self.video_files[index])
        frames = np.asarray(frames, dtype=np.float32)
        frames = np.transpose(frames, (0, 3, 1, 2))  # Change to (num_frames, channels, height, width)

        video_name = self.video_files[index].split('\\')[-1]
        return torch.from_numpy(frames), video_name

    def __len__(self):
        return self.count
