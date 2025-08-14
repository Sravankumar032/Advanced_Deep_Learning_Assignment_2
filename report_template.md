# Report: GAN vs WGAN vs SNGAN on MNIST (28×28)

**Author:** <Your Name Here>  
**Date:** <Fill Date>

---

## 1. Overview (what you did and why)
Briefly introduce GANs, the training instability problem, and the motivation for WGAN (Earth‑Mover distance, weight clipping) and SNGAN (spectral norm to control Lipschitz constant; hinge loss). State that you trained each model for **50+ epochs** on MNIST and compared training curves and samples.

## 2. Dataset
- **Dataset:** MNIST (60k train) resized/centered to 28×28, single channel.
- **Preprocessing:** Normalize to [-1, 1].
- **Batch size:** 128 (unless changed).

## 3. Model architectures
- **GAN:** DCGAN-style generator/discriminator (sigmoid/logits + BCE). Non‑saturating generator loss.
- **WGAN:** Same architecture but critic has no sigmoid; RMSprop, weight clipping (0.01); 5 critic steps per G step.
- **SNGAN:** Discriminator with **spectral normalization** on all conv/linear layers; **hinge loss**; Adam with β1=0.0, β2=0.9.

> Include param counts (printed at start of training).

## 4. Training setup
- **Epochs:** 50+
- **Optimizers:** As above. Learning rates per README.
- **Hardware:** <CPU/GPU info>
- **Fixed noise:** Used a fixed 64‑vector to create consistent sample grids across epochs.

## 5. Results
### 5.1 Sample grids
Paste 2–3 representative sample grids for each model (e.g., epoch 10, 30, 50). Comment on visual quality, diversity, and mode collapse (if any).

### 5.2 Loss curves
Insert `runs/<model>/plots/losses.png` for each model. Comment on trends:
- **GAN:** Oscillatory losses are normal; look for stable sample quality improvements.
- **WGAN:** Critic loss correlates with Wasserstein estimate; smoother progression helps stability.
- **SNGAN:** Often converges faster and more stably; hinge loss provides stronger gradients.

### 5.3 Discussion of training dynamics
- **GAN:** Susceptible to vanishing gradients if D saturates; non‑saturating trick helps.
- **WGAN:** Uses Earth‑Mover distance; weight clipping enforces 1‑Lipschitz but can underfit (too small clip) or explode (too large). Multiple critic steps stabilize training.
- **SNGAN:** Spectral norm provides principled Lipschitz control per-layer; hinge loss yields margins that stabilize learning.

### 5.4 Quality comparison (qualitative)
Rank the models based on your samples (e.g., SNGAN ≳ WGAN > GAN), noting sharpness and digit fidelity.

*(Optional bonus)* Compute a simple quantitative proxy, e.g., train a small MNIST classifier and report accuracy on generated samples per class, or compute MS‑SSIM diversity.

## 6. Limitations & future work
- Try **WGAN‑GP** for better Lipschitz enforcement.
- Use **class‑conditional** GANs for labeled control.
- Use **FID/IS** metrics with a dataset‑appropriate feature extractor.

## 7. Conclusion
Summarize what worked best and why, linking your observations to the theory behind each variant.

---

## Appendix: Exact commands run
```
<paste commands with seeds/hyperparameters>
```
