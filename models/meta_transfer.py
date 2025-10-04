import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMetaTransfer(nn.Module):
    """
    Causal Meta-Transfer module for few-shot adaptation.
    
    Computes causal importance vector w via meta-attention over base-class priors.
    Prunes non-causal dimensions using top-rho thresholding.
    
    Args:
        dc (int): Dimension of causal latent space (default: 64).
        num_base_classes (int): Number of base classes (e.g., 10 for MVTec-AD, 12 for VisA).
        alpha (float): Temperature scaling factor for attention (default: 5.0).
        rho (float): Sparsity ratio for pruning (default: 0.5).
    """
    def __init__(self, dc=64, num_base_classes=12, alpha=5.0, rho=0.5):
        super().__init__()
        self.dc = dc
        self.alpha = alpha
        self.rho = rho
        self.num_base_classes = num_base_classes
        
        # Base-class causal gradient priors: [num_base, dc]
        # Initialized as learnable parameters (can be loaded from pretraining)
        self.w_b = nn.Parameter(torch.randn(num_base_classes, dc))
        
        # Value projection matrix W_v (initialized as w_b)
        self.W_v = nn.Parameter(self.w_b.data.clone())
    
    def compute_gaussian_anomaly_likelihood(self, z_s_bar, mu_c, Sigma_c_inv, pi=1e-4):
        """
        Compute p(Y=1 | z_c) = sigmoid( -||z_c - mu_c||_{Sigma_c^{-1}}^2 + log(pi) )
        
        Args:
            z_s_bar (torch.Tensor): [dc] support prototype
            mu_c (torch.Tensor): [num_base, dc] class means
            Sigma_c_inv (torch.Tensor): [num_base, dc, dc] inverse covariances
            pi (float): Prior defect rate
        
        Returns:
            torch.Tensor: [num_base] likelihoods
        """
        diff = z_s_bar.unsqueeze(0) - mu_c  # [num_base, dc]
        mahal = torch.einsum('bi,bij,bj->b', diff, Sigma_c_inv, diff)  # [num_base]
        log_prob = -mahal + torch.log(torch.tensor(pi))
        return torch.sigmoid(log_prob)
    
    def compute_causal_gradient_prior(self, z_s_bar, mu_c, Sigma_c_inv, pi=1e-4):
        """
        Compute w_b^{(j)} = -2 * Sigma_c^{-1} * (z_s_bar - mu_c^{(j)}) * p(Y=1 | z_s_bar)
        
        Args:
            z_s_bar (torch.Tensor): [dc] support prototype
            mu_c (torch.Tensor): [num_base, dc] class means
            Sigma_c_inv (torch.Tensor): [num_base, dc, dc] inverse covariances
            pi (float): Prior defect rate
        
        Returns:
            torch.Tensor: [num_base, dc] causal gradient priors
        """
        p_y1 = self.compute_gaussian_anomaly_likelihood(z_s_bar, mu_c, Sigma_c_inv, pi)
        diff = z_s_bar.unsqueeze(0) - mu_c  # [num_base, dc]
        # Compute -2 * Sigma^{-1} * diff
        grad = -2 * torch.einsum('bij,bj->bi', Sigma_c_inv, diff)  # [num_base, dc]
        return grad * p_y1.unsqueeze(1)  # [num_base, dc]
    
    def forward(self, z_s_bar, mu_c, Sigma_c_inv):
        """
        Forward pass to compute pruned causal importance vector.
        
        Args:
            z_s_bar (torch.Tensor): [dc] support prototype
            mu_c (torch.Tensor): [num_base, dc] base-class means
            Sigma_c_inv (torch.Tensor): [num_base, dc, dc] base-class inverse covariances
        
        Returns:
            torch.Tensor: [dc] pruned causal importance vector (w * mask)
        """
        # Step 1: Compute causal gradient priors w_b^{(j)}
        w_b = self.compute_causal_gradient_prior(z_s_bar, mu_c, Sigma_c_inv)
        
        # Step 2: Meta-attention
        # Attention weights: softmax(alpha * <z_s_bar, w_b> / sqrt(dc))
        attn_logits = torch.matmul(z_s_bar.unsqueeze(0), w_b.t()) / (self.dc ** 0.5)  # [1, num_base]
        attn_weights = F.softmax(self.alpha * attn_logits, dim=-1)  # [1, num_base]
        
        # Step 3: Compute causal importance vector w
        w = torch.matmul(attn_weights, self.W_v)  # [1, dc]
        w = w.squeeze(0)  # [dc]
        
        # Step 4: Pruning via top-rho threshold
        k = int(self.rho * self.dc)
        _, topk_indices = torch.topk(w.abs(), k, sorted=False)
        mask = torch.zeros_like(w)
        mask[topk_indices] = 1.0
        
        return w * mask  # [dc]

    @torch.no_grad()
    def update_base_priors(self, mu_c, Sigma_c):
        """
        Update base-class priors from pretraining statistics.
        Call this after meta-pretraining to load learned normality manifold.
        
        Args:
            mu_c (torch.Tensor): [num_base, dc] class means
            Sigma_c (torch.Tensor): [num_base, dc, dc] class covariances
        """
        self.mu_c = mu_c
        self.Sigma_c = Sigma_c
        self.Sigma_c_inv = torch.inverse(Sigma_c + 1e-6 * torch.eye(self.dc, device=Sigma_c.device))
        # Optionally update w_b and W_v with gradient priors
        # For simplicity, we keep them as learnable parameters