# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class FluidSRDataset(Dataset):
    def __init__(self, h5_path='/home/diya/Projects/fluid-sr-thesis/data/raw/channel.h5', split='train', downsample_factor=2, y_slice=0):
        self.h5 = h5py.File(h5_path, 'r')
        self.keys = list(self.h5['velocity'].keys())  # e.g., '0' to '4'
        n_total = len(self.keys)
        if n_total == 0:
            raise ValueError("No velocity datasets found.")
        if split == 'train':
            self.keys = self.keys[:int(0.8 * n_total)]  # ~4 snapshots
        elif split == 'val':
            self.keys = self.keys[int(0.8 * n_total):int(0.9 * n_total)]  # ~0-1 snapshot
        else:
            self.keys = self.keys[int(0.9 * n_total):]  # ~0-1 snapshot
        if not self.keys:
            raise ValueError(f"No data for {split} split. Dataset has {n_total} samples.")
        self.downsample_factor = downsample_factor
        self.y_slice = y_slice  # Fixed y=0

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        vel = np.array(self.h5['velocity'][key])  # Shape: (64, 1, 64, 3) or (16, 16, 16, 3)
        u = vel[:, self.y_slice, :, 0]  # Shape: (64, 64) or (16, 16)
        w = vel[:, self.y_slice, :, 2]
        hr_data = np.sqrt(u**2 + w**2)
        hr = torch.tensor(hr_data, dtype=torch.float32)  # Shape: (64, 64) or (16, 16)
        lr = torch.nn.functional.avg_pool2d(
            hr.unsqueeze(0).unsqueeze(0), kernel_size=self.downsample_factor
        ).squeeze()  # Shape: (32, 32) or (8, 8)
        return lr, hr

    def close(self):
        self.h5.close()