# src/models/pifno.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = min(modes1, 32)  # Cap to 32 for 32x32 LR
        self.modes2 = min(modes2, 32)
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        _, _, h, w = x.shape
        out_ft = torch.zeros(batch_size, self.out_channels, h, w//2 + 1,
                             dtype=torch.cfloat, device=x.device)
        modes1 = min(self.modes1, h)
        modes2 = min(self.modes2, w//2 + 1)
        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
            x_ft[:, :, :modes1, :modes2], self.weights[:, :, :modes1, :modes2]
        )
        x = torch.fft.irfft2(out_ft, s=(h, w))
        return x

class PIFNO(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, modes1=32, modes2=32, width=32, scale_factor=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.scale_factor = scale_factor
        self.fc0 = nn.Linear(in_channels, self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.width)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(self.width, out_channels, kernel_size=1)

    def forward(self, x):
        batch_size, _, h, w = x.shape
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv0(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)
        x = self.final_conv(x)
        return x

    def physics_loss(self, pred, input, h5_path='/home/diya/Projects/fluid-sr-thesis/data/raw/channel.h5', key='0', y_slice=0):
        with h5py.File(h5_path, 'r') as f:
            u = np.array(f['velocity'][key][:, y_slice, :, 0])
            w = np.array(f['velocity'][key][:, y_slice, :, 2])
        u = torch.tensor(u, dtype=torch.float32, device=pred.device)
        w = torch.tensor(w, dtype=torch.float32, device=pred.device)
        dudx = torch.gradient(u, dim=0)[0]
        dwdz = torch.gradient(w, dim=1)[0]
        div = dudx + dwdz
        return torch.mean(div**2)