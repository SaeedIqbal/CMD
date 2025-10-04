# losses/vae_loss.py

import torch
import torch.nn.functional as F

def vae_loss(
    x_recon,
    x_true,
    mu_c,
    logvar_c,
    mu_n,
    logvar_n,
    beta_kl=0.5,
    recon_loss_type="l2"
):
    """
    Compute VAE loss for dual latent space (Z_c, Z_n).
    
    Total loss = Reconstruction Loss + beta_kl * (KL_c + KL_n)
    
    Args:
        x_recon (torch.Tensor): Reconstructed image [B, 3, H, W]
        x_true (torch.Tensor): Original image [B, 3, H, W]
        mu_c (torch.Tensor): Mean of causal latent [B, dc]
        logvar_c (torch.Tensor): Log-var of causal latent [B, dc]
        mu_n (torch.Tensor): Mean of non-causal latent [B, dn]
        logvar_n (torch.Tensor): Log-var of non-causal latent [B, dn]
        beta_kl (float): Weight for KL divergence (default: 0.5)
        recon_loss_type (str): 'l1' or 'l2' (default: 'l2')
    
    Returns:
        dict: {
            'total': total loss,
            'recon': reconstruction loss,
            'kl_c': KL divergence for Z_c,
            'kl_n': KL divergence for Z_n
        }
    """
    # Reconstruction loss
    if recon_loss_type == "l2":
        recon_loss = F.mse_loss(x_recon, x_true, reduction="mean")
    elif recon_loss_type == "l1":
        recon_loss = F.l1_loss(x_recon, x_true, reduction="mean")
    else:
        raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")
    
    # KL divergence for causal latent Z_c: D_KL(q(z_c|x) || p(z_c))
    kl_c = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
    kl_c = kl_c / x_true.size(0)  # Normalize by batch size
    
    # KL divergence for non-causal latent Z_n: D_KL(q(z_n|x) || p(z_n))
    kl_n = -0.5 * torch.sum(1 + logvar_n - mu_n.pow(2) - logvar_n.exp())
    kl_n = kl_n / x_true.size(0)
    
    # Total loss
    total_loss = recon_loss + beta_kl * (kl_c + kl_n)
    
    return {
        "total": total_loss,
        "recon": recon_loss,
        "kl_c": kl_c,
        "kl_n": kl_n
    }

def reparameterize(mu, logvar):
    """
    Reparameterization trick: z = mu + sigma * epsilon.
    
    Args:
        mu (torch.Tensor): Mean [B, d]
        logvar (torch.Tensor): Log variance [B, d]
    
    Returns:
        torch.Tensor: Sampled latent [B, d]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std