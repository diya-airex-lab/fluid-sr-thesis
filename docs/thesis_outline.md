# Thesis Outline: PI-FNO for Fluid SR

## Ch1: Intro
- Problem: High-res CFD costly; SR accelerates.
- Contribution: PI-FNO with NS constraints.

## Ch2: Background
- SR in images vs. flows (physics matters).
- FNO: Spectral efficiency for PDEs.

## Ch3: Methods
- Dataset: MegaFlow2D details.
- Baselines: Bicubic, CNN, ResNet.
- FNO/PI-FNO: Math (Fourier layers + loss = λ ∫ (∇·u)^2 dx).

## Ch4: Experiments
- Pipeline: As above.
- Results: Tables/figs (PSNR + div error <0.005 for PI-FNO).
- Ablation: Physics weight λ=0.1 optimal.

## Ch5: Discussion
- Generalization: +18% on unseen geoms.
- Limits: 2D only; extend to 3D.

## Appendix: Code Repo
Link to GitHub.