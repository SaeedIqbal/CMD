import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights

class ResNet101Backbone(nn.Module):
    """
    ResNet-101 backbone for industrial few-shot object detection.
    
    Extracts features from multiple stages (C2, C3, C4, C5) for:
    - Causal encoder (global features from C5)
    - RPN and RoIAlign (C4 for region proposals)
    - Optional multi-scale fusion (if needed in future)
    """
    def __init__(self, pretrained=True, return_stages=['C5']):
        """
        Args:
            pretrained (bool): Load ImageNet-pretrained weights.
            return_stages (list): List of stages to return (e.g., ['C4', 'C5']).
        """
        super().__init__()
        self.return_stages = return_stages
        
        # Load ResNet-101
        if pretrained:
            weights = ResNet101_Weights.IMAGENET1K_V1
            resnet = models.resnet101(weights=weights)
        else:
            resnet = models.resnet101(weights=None)
        
        # Stem: conv1 + bn1 + relu + maxpool
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # ResNet stages
        self.layer1 = resnet.layer1  # C2: /4
        self.layer2 = resnet.layer2  # C3: /8
        self.layer3 = resnet.layer3  # C4: /16
        self.layer4 = resnet.layer4  # C5: /32
        
        # Freeze batch norm if needed (optional)
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Optionally freeze early stages (not used in CMD, but kept for flexibility)."""
        pass  # CMD fine-tunes all layers
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [B, 3, H, W]
        
        Returns:
            dict: Features from requested stages.
                - 'C2': [B, 256, H/4, W/4]
                - 'C3': [B, 512, H/8, W/8]
                - 'C4': [B, 1024, H/16, W/16]
                - 'C5': [B, 2048, H/32, W/32]
        """
        features = {}
        
        x = self.stem(x)
        if 'C2' in self.return_stages:
            x = self.layer1(x)
            features['C2'] = x
        else:
            x = self.layer1(x)
        
        if 'C3' in self.return_stages:
            x = self.layer2(x)
            features['C3'] = x
        else:
            x = self.layer2(x)
        
        if 'C4' in self.return_stages:
            x = self.layer3(x)
            features['C4'] = x
        else:
            x = self.layer3(x)
        
        if 'C5' in self.return_stages:
            x = self.layer4(x)
            features['C5'] = x
        
        return features

# Convenience function for causal encoder (uses only C5)
def build_backbone(pretrained=True):
    """Build ResNet-101 backbone that returns only C5 features for causal encoder."""
    return ResNet101Backbone(pretrained=pretrained, return_stages=['C5'])

# Convenience function for detection head (uses C4 for RPN)
def build_detection_backbone(pretrained=True):
    """Build ResNet-101 backbone that returns C4 for RPN and C5 for RoI features."""
    return ResNet101Backbone(pretrained=pretrained, return_stages=['C4', 'C5'])