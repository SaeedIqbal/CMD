import torch
import torch.nn as nn
import torch.nn.functional as F

class InterventionalDetectionHead(nn.Module):
    """
    Faster R-CNN detection head with interventional similarity (S_do).
    
    Replaces standard support-query fusion with causal projection and S_do.
    
    Args:
        in_channels (int): Input feature channels (e.g., 2048 from ResNet-101 C5).
        hidden_dim (int): Hidden dimension for FC layers (default: 1024).
        num_classes (int): Number of classes (default: 2 for binary defect detection).
    """
    def __init__(self, in_channels=2048, hidden_dim=1024, num_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Shared feature transformation
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Classification head
        self.cls_score = nn.Linear(hidden_dim, num_classes)
        
        # Bounding box regression head
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform and zero bias for bbox."""
        for m in [self.fc1, self.fc2, self.cls_score]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Bbox regression: zero initialization
        nn.init.zeros_(self.bbox_pred.weight)
        nn.init.zeros_(self.bbox_pred.bias)
    
    def forward(self, query_feat, support_feat, w):
        """
        Forward pass with interventional similarity.
        
        Args:
            query_feat (torch.Tensor): [N, C] query RoI features
            support_feat (torch.Tensor): [K, C] support RoI features (K-shot)
            w (torch.Tensor): [C] causal importance vector (pruned)
        
        Returns:
            tuple: (cls_logits, bbox_deltas)
                - cls_logits: [N, num_classes]
                - bbox_deltas: [N, num_classes * 4]
        """
        # Apply causal projection: element-wise multiplication with w
        query_causal = query_feat * w.unsqueeze(0)  # [N, C]
        support_causal = support_feat * w.unsqueeze(0)  # [K, C]
        
        # Normalize for cosine similarity
        query_norm = F.normalize(query_causal, p=2, dim=1)  # [N, C]
        support_norm = F.normalize(support_causal, p=2, dim=1)  # [K, C]
        
        # Compute interventional similarity: [N, K]
        sim_matrix = torch.mm(query_norm, support_norm.t())  # [N, K]
        
        # Average over K support samples: [N, 1]
        sim = sim_matrix.mean(dim=1, keepdim=True)  # [N, 1]
        
        # Reweight support features: [N, C]
        # Broadcast sim to [N, C] and multiply with mean support feature
        mean_support = support_feat.mean(dim=0, keepdim=True)  # [1, C]
        weighted_support = sim * mean_support  # [N, C]
        
        # Fuse query and weighted support
        fused = query_feat + weighted_support  # [N, C]
        
        # Detection head
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        cls_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        
        return cls_logits, bbox_deltas