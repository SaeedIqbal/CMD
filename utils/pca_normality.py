import torch
import torch.nn as nn
import numpy as np

class PCANormalityManifold(nn.Module):
    """
    PCA-based normality manifold for causal latent space.
    
    Models N_c = { mu_c + U_c a | a in R^r } where:
        - mu_c: mean of normal causal latents
        - U_c: top-r principal components (dc x r)
        - r: PCA rank (e.g., 32 for 95% variance retention)
    
    Provides orthogonal projection: Pi_Nc(z) = mu_c + U_c U_c^T (z - mu_c)
    """
    def __init__(self, dc=64, r=32, device='cpu'):
        """
        Args:
            dc (int): Dimension of causal latent space.
            r (int): PCA rank (number of principal components).
            device (torch.device): Device for computation.
        """
        super().__init__()
        self.dc = dc
        self.r = r
        self.device = device
        
        # Parameters (will be set by fit())
        self.register_buffer('mu_c', None)      # [dc]
        self.register_buffer('U_c', None)       # [dc, r]
        self.is_fitted = False
    
    def fit(self, z_c_normal):
        """
        Fit PCA normality manifold from defect-free base data.
        
        Args:
            z_c_normal (torch.Tensor or np.ndarray): [N, dc] causal latents from normal images.
        """
        if isinstance(z_c_normal, torch.Tensor):
            z_c_normal = z_c_normal.detach().cpu().numpy()
        
        N, dc = z_c_normal.shape
        assert dc == self.dc, f"Expected {self.dc}D latents, got {dc}D"
        assert N > self.r, f"Need at least {self.r} normal samples for PCA (got {N})"
        
        # Compute mean
        mu_c = z_c_normal.mean(axis=0)  # [dc]
        
        # Center data
        z_centered = z_c_normal - mu_c  # [N, dc]
        
        # Compute covariance matrix
        cov = np.cov(z_centered, rowvar=False)  # [dc, dc]
        
        # Eigen decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Sort in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Select top-r components
        U_c = eigenvecs[:, :self.r]  # [dc, r]
        
        # Ensure right-handed coordinate system (optional)
        if np.linalg.det(U_c[:, :min(self.r, dc)]) < 0:
            U_c[:, 0] *= -1
        
        # Register buffers
        self.register_buffer('mu_c', torch.from_numpy(mu_c).float().to(self.device))
        self.register_buffer('U_c', torch.from_numpy(U_c).float().to(self.device))
        self.is_fitted = True
        
        # Return explained variance ratio for debugging
        explained_variance_ratio = eigenvals[:self.r] / eigenvals.sum()
        return explained_variance_ratio
    
    def project(self, z_c):
        """
        Orthogonal projection onto normality manifold: Pi_Nc(z_c).
        
        Args:
            z_c (torch.Tensor): [B, dc] causal latents.
        
        Returns:
            torch.Tensor: [B, dc] projected latents.
        """
        if not self.is_fitted:
            raise RuntimeError("Normality manifold not fitted. Call fit() first.")
        
        if z_c.device != self.mu_c.device:
            z_c = z_c.to(self.mu_c.device)
        
        # Center
        diff = z_c - self.mu_c.unsqueeze(0)  # [B, dc]
        
        # Project onto principal subspace
        proj = torch.mm(diff, self.U_c)      # [B, r]
        recon = torch.mm(proj, self.U_c.t()) # [B, dc]
        
        # Reconstruct in original space
        return self.mu_c.unsqueeze(0) + recon
    
    def deviation(self, z_c):
        """
        Compute L2 deviation from normality manifold.
        
        Args:
            z_c (torch.Tensor): [B, dc] causal latents.
        
        Returns:
            torch.Tensor: [B] L2 deviations.
        """
        z_proj = self.project(z_c)
        return torch.norm(z_c - z_proj, p=2, dim=1)
    
    def save(self, path):
        """Save manifold parameters to disk."""
        torch.save({
            'mu_c': self.mu_c,
            'U_c': self.U_c,
            'dc': self.dc,
            'r': self.r,
            'is_fitted': self.is_fitted
        }, path)
    
    def load(self, path):
        """Load manifold parameters from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.register_buffer('mu_c', checkpoint['mu_c'])
        self.register_buffer('U_c', checkpoint['U_c'])
        self.dc = checkpoint['dc']
        self.r = checkpoint['r']
        self.is_fitted = checkpoint['is_fitted']