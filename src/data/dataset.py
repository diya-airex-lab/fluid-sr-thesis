# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class FluidSRDataset(Dataset):
    def __init__(self, h5_path='data/raw/channel.h5', split='train', downsample_factor=2, y_slice=5):
        self.h5 = h5py.File(h5_path, 'r')
        self.keys = ['Velocity_0001']  # Hard-coded for current channel.h5
        n_total = len(self.keys)
        if n_total == 0:
            raise ValueError("No datasets found in channel.h5. Check HDF5 structure.")
        # Avoid splitting for single sample
        if n_total == 1:
            self.keys = self.keys
        else:
            if split == 'train':
                self.keys = self.keys[:int(0.8 * n_total)]
            elif split == 'val':
                self.keys = self.keys[int(0.8 * n_total):int(0.9 * n_total)]
            else:
                self.keys = self.keys[int(0.9 * n_total):]
        if not self.keys:
            raise ValueError(f"No data available for {split} split. Dataset has {n_total} samples.")
        self.downsample_factor = downsample_factor
        self.y_slice = y_slice

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        vel = np.array(self.h5[key])  # Shape: (10, 10, 10, 3)
        u = vel[:, self.y_slice, :, 0]  # x-z slice, u component (10x10)
        w = vel[:, self.y_slice, :, 2]  # w component (z-direction)
        hr_data = np.sqrt(u**2 + w**2)  # Velocity magnitude
        hr = torch.tensor(hr_data, dtype=torch.float32)  # Shape: (10, 10)
        lr = torch.nn.functional.avg_pool2d(
            hr.unsqueeze(0).unsqueeze(0), kernel_size=self.downsample_factor
        ).squeeze()  # Shape: (5, 5)
        return lr, hr

    def close(self):
        self.h5.close()