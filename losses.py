import torch
import torch.nn.functional as F

# Standard GAN losses (non-saturating G, BCE for D)
bce = torch.nn.BCEWithLogitsLoss()

def gan_d_loss(real_logits, fake_logits):
    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)
    loss_real = bce(real_logits, real_labels)
    loss_fake = bce(fake_logits, fake_labels)
    return loss_real + loss_fake

def gan_g_loss(fake_logits):
    real_labels = torch.ones_like(fake_logits)
    return bce(fake_logits, real_labels)

# WGAN losses (with weight clipping handled outside)
def wgan_d_loss(real_scores, fake_scores):
    # want to maximize real - fake; so minimize negative
    return -(real_scores.mean() - fake_scores.mean())

def wgan_g_loss(fake_scores):
    return -fake_scores.mean()

# Hinge losses (used in SNGAN)
def hinge_d_loss(real_scores, fake_scores):
    loss_real = torch.relu(1.0 - real_scores).mean()
    loss_fake = torch.relu(1.0 + fake_scores).mean()
    return loss_real + loss_fake

def hinge_g_loss(fake_scores):
    return -fake_scores.mean()