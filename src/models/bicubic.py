import torch
import torch.nn.functional as F

def bicubic_interpolate(lr_img, scale=4):
    """Bicubic upsampling baseline."""
    return F.interpolate(lr_img.unsqueeze(0), scale_factor=scale, mode='bicubic', align_corners=False).squeeze()

# def bicubic_interpolate(x):
#     return torch.nn.functional.interpolate(x, scale_factor=2, mode='bicubic')