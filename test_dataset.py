# test_dataset.py
from data.dataset import FluidSRDataset
try:
    dataset = FluidSRDataset('/home/diya/Projects/fluid-sr-thesis/data/raw/channel.h5', split='train')
    lr, hr = dataset[0]
    print(f"LR shape: {lr.shape}, HR shape: {hr.shape}")
    dataset.close()
except Exception as e:
    print(f"Error: {e}")