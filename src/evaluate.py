import argparse
from .data.dataset import FluidSRDataset
from .train import parser  # Reuse args
from .utils import compute_psnr, compute_ssim, plot_flow

# Add --ckpt arg
parser.add_argument('--ckpt', type=str, required=True)
args = parser.parse_args()

val_ds = FluidSRDataset(args.data_path, split='val')
val_loader = DataLoader(val_ds, batch_size=1)

model = globals()[args.model.title() + 'SR']()  # Load as above
model.load_state_dict(torch.load(args.ckpt))
model.eval()

psnrs, ssims = [], []
with torch.no_grad():
    for i, (lr, hr) in enumerate(val_loader):
        lr, hr = lr.cuda().unsqueeze(1), hr.cuda().unsqueeze(1)
        pred = model(lr)
        psnrs.append(compute_psnr(pred, hr))
        ssims.append(compute_ssim(pred[0], hr[0]))
        if i < 5:  # Viz first 5
            plot_flow(lr, hr, pred, f'results/{args.model}_viz_{i}.png')

print(f'{args.model} - Avg PSNR: {sum(psnrs)/len(psnrs):.2f}, SSIM: {sum(ssims)/len(ssims):.4f}')