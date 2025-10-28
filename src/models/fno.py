import torch
import torch.nn as nn
import torch.fft as fft

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1=12, modes2=12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights)
        x = fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes=16, width=64, layers=4):
        super().__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Linear(1, self.width)  # Input projection (scalar field)
        self.conv_layers = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes, self.modes) for _ in range(layers)])
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)  # Output scalar
        self.relu = nn.ReLU()

    def forward(self, x):  # x: (B, 1, LR_H, LR_W)
        # Embed LR grid
        grid = torch.stack(torch.meshgrid(torch.linspace(-1,1,x.shape[2]), torch.linspace(-1,1,x.shape[3])), dim=-1).to(x.device)
        x = torch.cat([x, grid], dim=1)  # Positional encoding
        x = x.view(x.shape[0], -1).unsqueeze(-1)  # Flatten for FC? Wait, adapt for 2D
        # Better: Treat as 2D field
        x = self.fc0(x.permute(0,2,3,1)).permute(0,3,1,2)  # (B, width, H, W)
        for conv in self.conv_layers:
            x = self.relu(conv(x) + x)  # Residual
        x = x.mean(dim=[2,3])  # Global avg pool for simplicity (or adapt)
        x = self.relu(self.fc1(x))
        x = self.fc2(x).view(x.shape[0], 1, *x.shape[0])  # Upsample post-process
        return F.interpolate(x, scale_factor=4, mode='bilinear')  # Post-upsample

# class FNOSR(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = torch.nn.Sequential(...)  # Placeholder
#     def forward(self, x):
#         return self.model(x)