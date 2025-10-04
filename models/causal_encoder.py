import torch
import torch.nn as nn
from .backbone import build_backbone

class CausalEncoder(nn.Module):
    """
    Dual-head encoder for Causal Meta-Disentanglement (CMD).
    
    Splits ResNet-101 features into:
        - Z_c (causal subspace): d_c = 64 dimensions (defect generators)
        - Z_n (non-causal subspace): d_n = 192 dimensions (nuisance factors)
    
    Enforces disentanglement via:
        - Orthogonality loss (L_ortho)
        - HSIC loss (statistical independence)
    """
    def __init__(self, dc=64, dn=192, pretrained=True):
        """
        Args:
            dc (int): Dimension of causal latent space (default: 64).
            dn (int): Dimension of non-causal latent space (default: 192).
            pretrained (bool): Use ImageNet-pretrained ResNet-101.
        """
        super().__init__()
        self.dc = dc
        self.dn = dn
        
        # Shared backbone (ResNet-101, returns C5 features)
        self.backbone = build_backbone(pretrained=pretrained)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Causal head: h_c: R^2048 -> R^dc
        self.h_c = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, dc)
        )
        
        # Non-causal head: h_n: R^2048 -> R^dn
        self.h_n = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, dn)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize linear layers with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input images [B, 3, H, W]
        
        Returns:
            tuple: (z_c, z_n)
                - z_c: [B, dc] causal latent codes
                - z_n: [B, dn] non-causal latent codes
        """
        # Extract C5 features [B, 2048, H/32, W/32]
        features = self.backbone(x)
        c5 = features['C5']  # dict from backbone
        
        # Global average pooling -> [B, 2048]
        feat = self.gap(c5).flatten(1)
        
        # Dual projection
        z_c = self.h_c(feat)  # [B, dc]
        z_n = self.h_n(feat)  # [B, dn]
        
        return z_c, z_n

    def encode_causal(self, x):
        """Encode only causal features (used in few-shot adaptation)."""
        z_c, _ = self.forward(x)
        return z_c

    def encode_noncausal(self, x):
        """Encode only non-causal features."""
        _, z_n = self.forward(x)
        return z_n