# src/train.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from models import *
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Ensure src/models/ contains __init__.py and model files (bicubic.py, cnn.py, resnet.py, fno.py, pifno.py)")
    sys.exit(1)
from data.dataset import FluidSRDataset
from utils import compute_psnr, compute_ssim

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='cnn', choices=['bicubic', 'cnn', 'resnet', 'fno', 'pifno'])
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1)  # Single sample for now
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--data_path', type=str, default='/home/diya/Projects/fluid-sr-thesis/data/raw/channel.h5')
args = parser.parse_args()

if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Dataset not found at {args.data_path}. Download 'channel.h5' from JHTDB.")

try:
    train_ds = FluidSRDataset(args.data_path, split='train')
except ValueError as e:
    print(f"Dataset error: {e}")
    sys.exit(1)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

if args.model == 'bicubic':
    model = lambda x: bicubic_interpolate(x)
else:
    model_class = globals().get(args.model.title() + 'SR') or PIFNO
    model = model_class(scale_factor=2)  # Pass scale_factor
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.model != 'bicubic' else None
criterion = torch.nn.MSELoss()

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for lr, hr in train_loader:
        lr, hr = lr.cuda().unsqueeze(1), hr.cuda().unsqueeze(1)
        pred = model(lr)
        loss = criterion(pred, hr)
        if args.model == 'pifno':
            phys_loss = model.physics_loss(pred, lr)  # Use model instance and pass lr
            loss += 0.1 * phys_loss
        if args.model != 'bicubic':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch}: Loss {total_loss / len(train_loader):.4f}')

if args.model != 'bicubic':
    os.makedirs('results', exist_ok=True)  # Create results directory if it doesn't exist
    torch.save(model.state_dict(), f'results/{args.model}.pth')