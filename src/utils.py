import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np

def compute_psnr(pred, target):
    return psnr(np.array(pred.cpu()), np.array(target.cpu()), data_range=1.0)

def compute_ssim(pred, target):
    return ssim(np.array(pred.cpu()), np.array(target.cpu()), data_range=1.0)

def divergence_error(u_pred, v_pred):  # Finite diff divergence
    dudx = (u_pred[:, :, 1:] - u_pred[:, :, :-1]).abs().mean()
    dvdy = (v_pred[:, :, 1:] - v_pred[:, :, :-1]).abs().mean()  # Simplified
    return (dudx + dvdy) / 2

# Viz helper
def plot_flow(lr, hr, pred, save_path):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(lr[0].cpu(), cmap='jet'); axs[0].set_title('LR')
    axs[1].imshow(hr[0].cpu(), cmap='jet'); axs[1].set_title('HR')
    axs[2].imshow(pred[0].cpu(), cmap='jet'); axs[2].set_title('Pred')
    plt.savefig(save_path)