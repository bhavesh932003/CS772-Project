# File: train.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_loader import load_ecg_windows
from autoencoder import AE
from vae import VAE, vae_loss
from losses import rho_loss

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
BETA = 0.1  # KL weighting factor
GRAD_CLIP = 1.0  # Gradient clipping threshold
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load and normalize ECG data
X, is_outlier = load_ecg_windows(
    data_dir="mitdb",
    window_size=250,
    stride=125
)
N_SAMPLES, INPUT_DIM = X.shape

# 2. Prepare DataLoader
tensor_X = torch.from_numpy(X)
dataset = TensorDataset(tensor_X, tensor_X)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Updated evaluation utilities
def evaluate_model(model, data_tensor):
    model.eval()
    with torch.no_grad():
        recon = model(data_tensor.to(DEVICE))
    recon = recon.cpu().numpy()
    orig = data_tensor.numpy()
    mse = ((recon - orig) ** 2).mean(axis=1)
    
    # Dynamic RHO-epsilon (1% of max MSE)
    rho_epsilon = 0.01 * mse.max()
    rho_scores = mse / (mse + rho_epsilon + 1e-9)
    return mse, rho_scores

def evaluate_vae_per_sample(model, data_tensor):
    model.eval()
    all_recon, all_mu, all_logvar = [], [], []

    full_loader = DataLoader(TensorDataset(data_tensor, data_tensor),
                             batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for x_batch, _ in full_loader:
            x = x_batch.to(DEVICE)
            recon, mu, logvar = model(x)
            all_recon.append(recon.cpu())
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    recon = torch.cat(all_recon)
    mu = torch.cat(all_mu)
    logvar = torch.cat(all_logvar)

    # Mean reconstruction error per dimension
    recon_err = ((recon - data_tensor) ** 2).mean(dim=1).numpy()
    kld = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).numpy()
    elbo = recon_err + BETA * kld  # Scaled ELBO
    
    return recon_err, kld, elbo

# 4. Initialize models
HIDDEN_DIM = 128
LATENT_DIM = 32

ae = AE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
rho_ae = AE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
vae = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)

# 5. Training loops with gradient fixes
def train_ae(model, optimizer, loss_fn, model_name):
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x_batch, _ in loader:
            x = x_batch.to(DEVICE)
            recon = model(x)
            loss = loss_fn(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / N_SAMPLES
        print(f"[{model_name}] Epoch {epoch}/{EPOCHS}  Avg Loss: {avg_loss:.6f}")

# Train Vanilla AE
opt_ae = torch.optim.Adam(ae.parameters(), lr=LR)
train_ae(ae, opt_ae, F.mse_loss, "AE")

# Train RHO-AE
opt_rho = torch.optim.Adam(rho_ae.parameters(), lr=LR)
train_ae(rho_ae, opt_rho, rho_loss, "RHO")

# Train VAE with gradient handling
opt_vae = torch.optim.Adam(vae.parameters(), lr=LR)
for epoch in range(1, EPOCHS + 1):
    vae.train()
    total_elbo = 0.0
    recon_losses = []
    kld_losses = []
    
    for x_batch, _ in loader:
        x = x_batch.to(DEVICE)
        recon, mu, logvar = vae(x)
        
        # Calculate loss (single tensor now)
        total_loss = vae_loss(recon, x, mu, logvar)
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Apply β-weighting and normalize
        weighted_loss = (recon_loss + BETA * kld_loss) / x.size(0)
        
        opt_vae.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), GRAD_CLIP)
        opt_vae.step()
        
        # For logging
        total_elbo += total_loss.item()
        recon_losses.append(recon_loss.item() / x.size(0))
        kld_losses.append(kld_loss.item() / x.size(0))
    
    avg_recon = np.mean(recon_losses)
    avg_kld = np.mean(kld_losses)
    print(
        f"[VAE] Epoch {epoch}/{EPOCHS}  "
        f"Total ELBO: {total_elbo:.1f}  "
        f"Recon: {avg_recon:.3f}  "
        f"KLD: {avg_kld:.3f}"
    )

# 6. Evaluation and metrics export
ae_mse, _ = evaluate_model(ae, tensor_X)
_, rho_scores = evaluate_model(rho_ae, tensor_X)
vae_recon, vae_kld, vae_elbo = evaluate_vae_per_sample(vae, tensor_X)

# Save results
df = pd.DataFrame({
    'sample_id': np.arange(N_SAMPLES),
    'is_outlier': is_outlier,
    'ae_mse': ae_mse,
    'rho_score': rho_scores,
    'vae_recon': vae_recon,
    'vae_kld': vae_kld,
    'vae_elbo': vae_elbo,
})
os.makedirs('results', exist_ok=True)
df.to_csv('results/metrics.csv', index=False)

# Post-training diagnostics
if df['vae_elbo'].max() > 1000:
    outliers = df[df['vae_elbo'] > 1000]
    print(f"\n⚠️ Extreme ELBO values detected in {len(outliers)} samples!")
else:
    print("\n✅ All ELBO values within expected range")

print("Training complete. Metrics saved to results/metrics.csv")