import torch
import torch.nn.functional as F

# RHO-Loss: original MSE-based
def rho_loss(recon, x, rho=1e-3):
    # per-sample MSE
    err2 = F.mse_loss(recon, x, reduction='none').mean(dim=1)
    # RHO-based weighting
    return (err2 / (err2 + rho)).mean()

# Variant: L1-based RHO
def rho_loss_l1(recon, x, rho=1e-3):
    err1 = F.l1_loss(recon, x, reduction='none').mean(dim=1)
    return (err1 / (err1 + rho)).mean()