# losses/ortho_loss.py

import torch
import torch.nn as nn

def ortho_loss(z_c, z_n):
    """
    Compute orthogonality loss between causal and non-causal latents.
    
    L_ortho = || (1/B) * sum_{b=1}^B h_c(x_b)^T h_n(x_b) ||_F^2
    
    This loss enforces that the column spaces of Z_c and Z_n are orthogonal,
    promoting disentanglement at the feature level.
    
    Args:
        z_c (torch.Tensor): [B, dc] causal latent codes
        z_n (torch.Tensor): [B, dn] non-causal latent codes
    
    Returns:
        torch.Tensor: Scalar orthogonality loss
    """
    B = z_c.size(0)
    if B == 0:
        return torch.tensor(0.0, device=z_c.device)
    
    # Compute batch-wise inner product: [dc, dn]
    # (1/B) * Z_c^T Z_n
    gram_matrix = torch.mm(z_c.t(), z_n) / B  # [dc, dn]
    
    # Frobenius norm squared
    loss = torch.norm(gram_matrix, p='fro') ** 2
    
    return loss

class OrthoLoss(nn.Module):
    """
    Orthogonality loss module for easy integration into training loops.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, z_c, z_n):
        return ortho_loss(z_c, z_n)