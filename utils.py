import os, json, math, random, time
from pathlib import Path
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import matplotlib.pyplot as plt

def make_dirs(run_dir: str):
    samples = Path(run_dir)/"samples"
    ckpts = Path(run_dir)/"ckpts"
    plots = Path(run_dir)/"plots"
    for p in [samples, ckpts, plots]:
        p.mkdir(parents=True, exist_ok=True)
    return {"samples": samples, "ckpts": ckpts, "plots": plots}

def get_dataloader(batch_size: int, num_workers: int = 2) -> DataLoader:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST to [-1,1]
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

def save_grid(tensor, path, nrow=8):
    vutils.save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))

def fixed_noise(nz: int, batch: int = 64, device: str = "cpu"):
    torch.manual_seed(42)
    return torch.randn(batch, nz, 1, 1, device=device)

def plot_losses(logs: Dict[str, list], out_path: str):
    plt.figure()
    for k, v in logs.items():
        if isinstance(v, list) and len(v) > 0:
            plt.plot(v, label=k)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def dump_logs(logs: Dict[str, Any], out_path: str):
    with open(out_path, "w") as f:
        json.dump(logs, f, indent=2)

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)