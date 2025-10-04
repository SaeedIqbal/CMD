import torch
import torch.nn as nn

def rbf_kernel(X, Y=None, sigma=None):
    """
    Compute RBF (Gaussian) kernel matrix.
    
    Args:
        X (torch.Tensor): [N, D] input features
        Y (torch.Tensor, optional): [M, D] input features (if None, Y=X)
        sigma (float, optional): Kernel bandwidth. If None, use median heuristic.
    
    Returns:
        torch.Tensor: [N, M] kernel matrix
    """
    if Y is None:
        Y = X
    
    # Compute squared Euclidean distance matrix
    X_norm = (X ** 2).sum(1).view(-1, 1)  # [N, 1]
    Y_norm = (Y ** 2).sum(1).view(1, -1)  # [1, M]
    dist_sq = X_norm + Y_norm - 2.0 * torch.mm(X, Y.t())  # [N, M]
    
    # Median heuristic for sigma
    if sigma is None:
        # Compute median of pairwise distances
        dist_vec = dist_sq.view(-1)
        dist_vec = dist_vec[dist_vec > 0]  # Remove zero distances (diagonal)
        if len(dist_vec) == 0:
            sigma = 1.0
        else:
            sigma = torch.sqrt(0.5 * torch.median(dist_vec))
    
    # Compute RBF kernel
    K = torch.exp(-dist_sq / (2.0 * sigma ** 2))
    return K

def hsic_loss(z_c, z_n, sigma_c=None, sigma_n=None):
    """
    Compute HSIC loss between causal and non-causal latents.
    
    HSIC(Z_c, Z_n) = (1/(m-1)^2) * Tr(K_c H K_n H)
    where H = I - (1/m) 11^T is the centering matrix.
    
    Args:
        z_c (torch.Tensor): [B, dc] causal latent codes
        z_n (torch.Tensor): [B, dn] non-causal latent codes
        sigma_c (float, optional): RBF kernel bandwidth for Z_c
        sigma_n (float, optional): RBF kernel bandwidth for Z_n
    
    Returns:
        torch.Tensor: Scalar HSIC loss
    """
    B = z_c.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=z_c.device)
    
    # Compute kernel matrices
    K_c = rbf_kernel(z_c, sigma=sigma_c)  # [B, B]
    K_n = rbf_kernel(z_n, sigma=sigma_n)  # [B, B]
    
    # Centering matrix: H = I - (1/B) * 11^T
    H = torch.eye(B, device=z_c.device) - 1.0 / B
    
    # Compute HSIC: Tr(K_c H K_n H)
    # Use: Tr(AB) = sum(A * B^T)
    K_c_H = torch.mm(K_c, H)
    K_n_H = torch.mm(K_n, H)
    hsic = torch.trace(torch.mm(K_c_H, K_n_H)) / ((B - 1) ** 2)
    
    return hsic

class HSICLoss(nn.Module):
    """
    HSIC loss module for easy integration into training loops.
    
    Args:
        sigma_c (float, optional): Kernel bandwidth for Z_c
        sigma_n (float, optional): Kernel bandwidth for Z_n
    """
    def __init__(self, sigma_c=None, sigma_n=None):
        super().__init__()
        self.sigma_c = sigma_c
        self.sigma_n = sigma_n
    
    def forward(self, z_c, z_n):
        return hsic_loss(z_c, z_n, self.sigma_c, self.sigma_n)