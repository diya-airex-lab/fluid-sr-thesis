import torch
import torch.nn as nn

class UNSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, scale=4):
        super().__init__()
        self.scale = scale
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        # Decoder
        self.dec1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Upsample LR to match HR size roughly
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        # Encode
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        # Decode
        d1 = self.relu(self.dec1(e2))
        out = self.dec2(d1)
        return out

# src/models/cnn.py
# class CNNSR(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = torch.nn.Sequential(...)  # Placeholder
#     def forward(self, x):
#         return self.model(x)