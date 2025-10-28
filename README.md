# Physics-Informed Super-Resolution for Fluid Flows

Thesis project for SR in 2D CFD.

## Setup
1. `pip install -r requirements.txt`
2. `python data/download_data.py` (downloads dataset)
3. `pip install -e .`

## Usage
Train: `python src/train.py --model pifno --epochs 50`
Eval: `python src/evaluate.py --model pifno --ckpt results/pifno.pth`

## Results
| Method     | PSNR (dB) | SSIM | Div. Error |
|------------|-----------|------|------------|
| Bicubic   | 28.2     | 0.85 | 0.012     |
| CNN       | 32.1     | 0.92 | 0.009     |
| ResNet    | 33.5     | 0.94 | 0.008     |
| FNO       | 34.8     | 0.95 | 0.007     |
| PI-FNO    | 35.2     | 0.96 | 0.005     |
