# losses/normality_deviation.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hsic import hsic_loss

class NormalityDeviationLoss(nn.Module):
    """
    Normality Deviation Loss for meta-pretraining on synthetic corruption tasks.
    
    L_dev = ||Z_c - Pi_Nc(Z_c)||_2 + lambda * HSIC(Z_c, Z_n) + mu * R_smooth(Z_c)
    
    Args:
        lambda_hsic (float): Weight for HSIC loss (default: 1.2).
        mu_smooth (float): Weight for smoothness regularization (default: 0.05).
        r_pca (int): Rank for PCA-based normality manifold (default: 32).
        device (torch.device): Device for computation.
    """
    def __init__(self, lambda_hsic=1.2, mu_smooth=0.05, r_pca=32, device='cpu'):
        super().__init__()
        self.lambda_hsic = lambda_hsic
        self.mu_smooth = mu_smooth
        self.r_pca = r_pca
        self.device = device
        
        # Normality manifold parameters (to be initialized from base data)
        self.register_buffer('mu_c', None)      # [dc]
        self.register_buffer('U_c', None)       # [dc, r]
        self.is_fitted = False
    
    def fit_normality_manifold(self, z_c_normal):
        """
        Fit the normality manifold N_c = {mu_c + U_c a | a in R^r} using PCA.
        
        Args:
            z_c_normal (torch.Tensor): [N, dc] causal latents from normal images.
        """
        N, dc = z_c_normal.shape
        assert N > self.r_pca, f"Need at least {self.r_pca} normal samples for PCA"
        
        # Compute mean
        mu_c = z_c_normal.mean(dim=0)  # [dc]
        
        # Center data
        z_centered = z_c_normal - mu_c  # [N, dc]
        
        # PCA via SVD
        U, S, Vt = torch.svd(z_centered)
        U_c = Vt[:self.r_pca].t()  # [dc, r]
        
        # Register buffers
        self.register_buffer('mu_c', mu_c.to(self.device))
        self.register_buffer('U_c', U_c.to(self.device))
        self.is_fitted = True
    
    def project_onto_manifold(self, z_c):
        """
        Compute Pi_Nc(z_c) = mu_c + U_c U_c^T (z_c - mu_c)
        
        Args:
            z_c (torch.Tensor): [B, dc] causal latents.
        
        Returns:
            torch.Tensor: [B, dc] projected latents.
        """
        if not self.is_fitted:
            raise RuntimeError("Normality manifold not fitted. Call fit_normality_manifold() first.")
        
        diff = z_c - self.mu_c.unsqueeze(0)  # [B, dc]
        proj = torch.mm(diff, self.U_c)      # [B, r]
        recon = torch.mm(proj, self.U_c.t()) # [B, dc]
        return self.mu_c.unsqueeze(0) + recon
    
    def spatial_smoothness_loss(self, z_c_map):
        """
        Enforce spatial smoothness in latent feature maps.
        
        Args:
            z_c_map (torch.Tensor): [B, dc, H, W] causal feature maps.
        
        Returns:
            torch.Tensor: Scalar smoothness loss.
        """
        # Compute gradients
        grad_x = z_c_map[:, :, :, 1:] - z_c_map[:, :, :, :-1]  # [B, dc, H, W-1]
        grad_y = z_c_map[:, :, 1:, :] - z_c_map[:, :, :-1, :]  # [B, dc, H-1, W]
        
        # Sum of squared gradients
        smooth_loss = (grad_x ** 2).sum() + (grad_y ** 2).sum()
        return smooth_loss / z_c_map.numel()
    
    def forward(self, z_c, z_n, z_c_map=None):
        """
        Compute normality deviation loss.
        
        Args:
            z_c (torch.Tensor): [B, dc] causal latents (after GAP).
            z_n (torch.Tensor): [B, dn] non-causal latents.
            z_c_map (torch.Tensor, optional): [B, dc, H, W] causal feature maps for smoothness.
        
        Returns:
            dict: {
                'total': total loss,
                'deviation': ||Z_c - Pi_Nc(Z_c)||_2,
                'hsic': HSIC(Z_c, Z_n),
                'smooth': R_smooth(Z_c) (if z_c_map provided)
            }
        """
        if not self.is_fitted:
            raise RuntimeError("Normality manifold not fitted. Call fit_normality_manifold() first.")
        
        # 1. Deviation from normality manifold
        z_c_proj = self.project_onto_manifold(z_c)  # [B, dc]
        deviation_loss = F.mse_loss(z_c, z_c_proj)
        
        # 2. HSIC loss for disentanglement
        hsic = hsic_loss(z_c, z_n)
        
        # 3. Spatial smoothness (optional)
        smooth_loss = torch.tensor(0.0, device=z_c.device)
        if z_c_map is not None:
            smooth_loss = self.spatial_smoothness_loss(z_c_map)
        
        # Total loss
        total_loss = deviation_loss + self.lambda_hsic * hsic + self.mu_smooth * smooth_loss
        
        return {
            'total': total_loss,
            'deviation': deviation_loss,
            'hsic': hsic,
            'smooth': smooth_loss
        }