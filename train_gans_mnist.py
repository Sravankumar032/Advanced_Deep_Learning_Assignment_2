import os, argparse, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import make_grid
from utils import get_dataloader, save_grid, make_dirs, fixed_noise, plot_losses, dump_logs, count_params
from models import Generator, Discriminator
from losses import gan_d_loss, gan_g_loss, wgan_d_loss, wgan_g_loss, hinge_d_loss, hinge_g_loss

def parse_args():
    p = argparse.ArgumentParser(description="Train GAN/WGAN/SNGAN on MNIST 28x28")
    p.add_argument("--model", choices=["gan", "wgan", "sngan"], default="gan")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--lrG", type=float, default=2e-4)
    p.add_argument("--lrD", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--opt", choices=["adam","rmsprop"], default="adam")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--critic-iters", type=int, default=5, help="for WGAN")
    p.add_argument("--clip", type=float, default=0.01, help="weight clipping for WGAN")
    p.add_argument("--hinge", action="store_true", help="use hinge loss (SNGAN)")
    p.add_argument("--run-dir", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    device = args.device
    run_name = args.model
    run_dir = args.run_dir or f"runs/{run_name}"
    dirs = make_dirs(run_dir)
    dataloader = get_dataloader(args.batch_size)

    # Models
    use_sigmoid = (args.model == "gan")
    spectral = (args.model == "sngan")
    netG = Generator(nz=args.nz, ngf=args.ngf).to(device)
    netD = Discriminator(ndf=args.ndf, use_sigmoid=False if spectral or args.model=="wgan" else use_sigmoid,
                         spectral=spectral).to(device)

    print(f"Params - G: {count_params(netG):,} | D: {count_params(netD):,}")
    # Optims
    if args.model == "wgan" or args.opt == "rmsprop":
        optD = optim.RMSprop(netD.parameters(), lr=args.lrD)
        optG = optim.RMSprop(netG.parameters(), lr=args.lrG)
    else:
        beta1 = 0.0 if (args.model=="sngan" and args.hinge) else args.beta1
        beta2 = 0.9 if (args.model=="sngan" and args.hinge) else args.beta2
        optD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(beta1, beta2))
        optG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(beta1, beta2))

    fixed_z = fixed_noise(args.nz, 64, device=device)

    logs = {"d_loss": [], "g_loss": []}
    step = 0

    for epoch in range(1, args.epochs + 1):
        for i, (real, _) in enumerate(dataloader):
            real = real.to(device)
            bsz = real.size(0)
            z = torch.randn(bsz, args.nz, 1, 1, device=device)

            # ---------------------- Discriminator / Critic ----------------------
            if args.model == "wgan":
                # multiple critic iterations per G step
                for _ in range(args.critic_iters):
                    optD.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        fake = netG(z).detach()
                    real_scores = netD(real)
                    fake_scores = netD(fake)
                    d_loss = wgan_d_loss(real_scores, fake_scores)
                    d_loss.backward()
                    optD.step()
                    # weight clipping
                    for p in netD.parameters():
                        p.data.clamp_(-args.clip, args.clip)
                logs["d_loss"].append(d_loss.item())
            else:
                # one D step
                optD.zero_grad(set_to_none=True)
                fake = netG(z).detach()
                real_logits = netD(real)
                fake_logits = netD(fake)
                if args.model == "sngan" and args.hinge:
                    d_loss = hinge_d_loss(real_logits, fake_logits)
                else:
                    d_loss = gan_d_loss(real_logits, fake_logits)
                d_loss.backward()
                optD.step()
                logs["d_loss"].append(d_loss.item())

            # ------------------------------- Generator --------------------------
            optG.zero_grad(set_to_none=True)
            z = torch.randn(bsz, args.nz, 1, 1, device=device)
            fake = netG(z)
            if args.model == "wgan":
                fake_scores = netD(fake)
                g_loss = wgan_g_loss(fake_scores)
            elif args.model == "sngan" and args.hinge:
                fake_scores = netD(fake)
                g_loss = hinge_g_loss(fake_scores)
            else:
                fake_logits = netD(fake)
                g_loss = gan_g_loss(fake_logits)
            g_loss.backward()
            optG.step()
            logs["g_loss"].append(g_loss.item())
            step += 1

        # ------------------ end epoch: save samples & plots ------------------
        if epoch % args.sample_every == 0:
            with torch.no_grad():
                fake = netG(fixed_z).cpu()
            grid_path = Path(dirs["samples"])/f"epoch_{epoch:03d}.png"
            save_grid(fake, grid_path, nrow=8)
            plot_losses(logs, Path(dirs["plots"])/"losses.png")
            dump_logs(logs, Path(run_dir)/"logs.json")
            torch.save(netG.state_dict(), Path(dirs["ckpts"])/f"G_epoch{epoch:03d}.pt")
            torch.save(netD.state_dict(), Path(dirs["ckpts"])/f"D_epoch{epoch:03d}.pt")
            print(f"[Epoch {epoch}] saved samples and checkpoints.")

    print("Training finished.")

if __name__ == "__main__":
    main()
