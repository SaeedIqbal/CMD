import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDecoder(nn.Module):
    """
    U-Net-style decoder for image reconstruction from causal and non-causal latents.
    
    Reconstructs image X_hat = g(Z_c, Z_n) + eta, where:
        - Z_c: [B, dc] causal latent codes
        - Z_n: [B, dn] non-causal latent codes
        - Output: [B, 3, H, W] reconstructed image
    
    Architecture:
        - Latent fusion → 4x4 feature map
        - 5 upsampling blocks with skip connections (from ResNet-101 backbone)
        - Final 1x1 conv to 3 channels
    """
    def __init__(self, dc=64, dn=192, backbone_channels=[256, 512, 1024, 2048], out_channels=3):
        """
        Args:
            dc (int): Dimension of causal latent space.
            dn (int): Dimension of non-causal latent space.
            backbone_channels (list): Channels from ResNet-101 stages [C2, C3, C4, C5].
            out_channels (int): Number of output channels (3 for RGB).
        """
        super().__init__()
        self.dc = dc
        self.dn = dn
        self.backbone_channels = backbone_channels
        
        # Initial latent projection to 4x4 feature map
        self.latent_proj = nn.Sequential(
            nn.Linear(dc + dn, 2048 * 4 * 4),
            nn.ReLU()
        )
        
        # Upsampling blocks with skip connections
        self.up4 = UpBlock(2048, 1024, 1024)  # C5 → C4
        self.up3 = UpBlock(1024, 512, 512)    # C4 → C3
        self.up2 = UpBlock(512, 256, 256)     # C3 → C2
        self.up1 = UpBlock(256, 64, 64)       # C2 → 1/2 resolution
        
        # Final upsampling to full resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] (to match normalized inputs)
        )
    
    def forward(self, z_c, z_n, skip_features=None):
        """
        Forward pass.
        
        Args:
            z_c (torch.Tensor): Causal latents [B, dc]
            z_n (torch.Tensor): Non-causal latents [B, dn]
            skip_features (dict, optional): Skip connections from backbone
                - 'C2': [B, 256, H/4, W/4]
                - 'C3': [B, 512, H/8, W/8]
                - 'C4': [B, 1024, H/16, W/16]
                - 'C5': [B, 2048, H/32, W/32] (not used)
        
        Returns:
            torch.Tensor: Reconstructed image [B, 3, H, W]
        """
        # Fuse latents
        z = torch.cat([z_c, z_n], dim=1)  # [B, dc+dn]
        x = self.latent_proj(z)           # [B, 2048*4*4]
        x = x.view(-1, 2048, 4, 4)        # [B, 2048, 4, 4]
        
        # Upsample with skip connections
        if skip_features is not None:
            x = self.up4(x, skip_features['C4'])  # [B, 1024, 8, 8]
            x = self.up3(x, skip_features['C3'])  # [B, 512, 16, 16]
            x = self.up2(x, skip_features['C2'])  # [B, 256, 32, 32]
        else:
            # Without skip connections (for meta-pretraining on normal images)
            x = self.up4(x, None)
            x = self.up3(x, None)
            x = self.up2(x, None)
        
        x = self.up1(x, None)             # [B, 64, 64, 64]
        x = self.final_up(x)              # [B, 3, H, W]
        return x


class UpBlock(nn.Module):
    """U-Net upsampling block with optional skip connection."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        ) if skip_channels > 0 else nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.use_skip = skip_channels > 0
    
    def forward(self, x, skip=None):
        x = self.up(x)  # Upsample
        if self.use_skip and skip is not None:
            # Crop skip to match x size (if needed)
            diff_h = skip.size(2) - x.size(2)
            diff_w = skip.size(3) - x.size(3)
            skip = skip[:, :, diff_h//2:-(diff_h-diff_h//2), diff_w//2:-(diff_w-diff_w//2)]
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)