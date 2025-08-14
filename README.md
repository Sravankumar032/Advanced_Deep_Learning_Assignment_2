# Advanced_Deep_Learning_Assignment_2
This Repository contains a **complete, end‑to‑end assignment** to implement and compare
# GAN vs WGAN vs SNGAN — MNIST (28×28)

- **GAN (DCGAN-style)**
- **WGAN (weight clipping)**
- **SNGAN (Spectral Normalization + Hinge loss)**

Trained each model for **50+ epochs** on MNIST and compared **training dynamics** and **sample quality**.

---
Detailed Steps to Implement 

### 1) Create & activate environment
```bash
python -m venv .venv
```

### 2) Install deps
```bash
pip install -r requirements.txt
```

### 3) Train any model
```bash
# GAN (non-saturating with BCE)
python train_gans_mnist.py --model gan --epochs 50

# WGAN (critic with weight clipping, RMSprop)
python train_gans_mnist.py --model wgan --epochs 50 --critic-iters 5 --clip 0.01 --opt rmsprop --lrD 5e-5 --lrG 5e-5

# SNGAN (spectral norm + hinge loss)
python train_gans_mnist.py --model sngan --epochs 50 --hinge
```

### 4) Where outputs go
- **`runs/<model>/samples/`** → image grids every few epochs (quality check)
- **`runs/<model>/ckpts/`** → generator/discriminator checkpoints
- **`runs/<model>/plots/`** → loss curves
- **`runs/<model>/logs.json`** → numeric logs you can analyze later

### 5) Compare models
After training all three, open **`report_template.md`**, drop in your sample grids, and perfrom analysis.

---

## Suggested hyperparameters

- **GAN**
  - Optimizer: Adam (β1=0.5, β2=0.999), lr=2e-4
  - Batch size: 128
  - Non-saturating generator loss, BCE for discriminator

- **WGAN**
  - Optimizer: RMSprop, lr=5e-5
  - Critic steps per G step: 5
  - Weight clipping: 0.01
  - No sigmoid on critic output

- **SNGAN**
  - Spectral Norm on **all conv/linear** layers of the discriminator
  - Hinge loss (D: max(0,1-D(x)) + max(0,1+D(G(z))); G: -E[D(G(z))])
  - Optimizer: Adam, lr=2e-4, β1=0.0, β2=0.9

---

---

## Files
- `train_gans_mnist.py` – training loop + CLI
- `models.py` – Generator/Discriminator for GAN/WGAN (DCGAN-ish) and SNGAN variants
- `losses.py` – loss functions for GAN, WGAN, SNGAN (hinge)
- `utils.py` – data loader, fixed noise, grid saving, logging & plotting
- `report_template.md` – fill this for your submission

---

