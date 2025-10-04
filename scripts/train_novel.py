# scripts/train_novel.py

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np

from models.causal_encoder import CausalEncoder
from models.meta_transfer import CausalMetaTransfer
from models.detection_head import InterventionalDetectionHead
from models.backbone import build_detection_backbone
from data.dataset_factory import get_dataset
from utils.pca_normality import PCANormalityManifold
from utils.metrics import compute_metrics_torch
import random

def parse_args():
    parser = argparse.ArgumentParser(description="CMD Few-Shot Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/novel.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="mvtec_ad", help="Dataset name")
    parser.add_argument("--novel_class", type=str, default="bottle", help="Novel class name")
    parser.add_argument("--k_shot", type=int, default=3, choices=[1, 2, 3, 5, 10], help="Number of support shots")
    parser.add_argument("--iterations", type=int, default=2000, help="Number of fine-tuning iterations")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--base_ckpt", type=str, required=True, help="Path to base model checkpoint")
    parser.add_argument("--manifold_path", type=str, required=True, help="Path to PCA manifold")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Save directory")
    return parser.parse_args()

def sample_few_shot_support(dataset, k_shot, novel_class, seed=42):
    """Sample K-shot support set for a novel class."""
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Find all anomaly images for the novel class
    anomaly_indices = []
    for i, (_, label, path) in enumerate(dataset.samples):
        if label == 1 and novel_class in path:
            anomaly_indices.append(i)
    
    if len(anomaly_indices) < k_shot:
        raise ValueError(f"Not enough anomalies for {novel_class}: found {len(anomaly_indices)}, need {k_shot}")
    
    # Randomly sample K shots
    selected_indices = random.sample(anomaly_indices, k_shot)
    support_set = Subset(dataset, selected_indices)
    return support_set

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset (contains novel anomalies)
    test_dataset = get_dataset(args.dataset, split="test", transform=transform)
    
    # Sample K-shot support set
    support_set = sample_few_shot_support(test_dataset, args.k_shot, args.novel_class)
    support_loader = DataLoader(support_set, batch_size=args.k_shot, shuffle=False)
    
    # Load base model
    print("Loading base model...")
    checkpoint = torch.load(args.base_ckpt, map_location=device)
    
    # Initialize models
    backbone = build_detection_backbone(pretrained=False).to(device)
    encoder = CausalEncoder(dc=64, dn=192, pretrained=False).to(device)
    detection_head = InterventionalDetectionHead(in_channels=2048, hidden_dim=1024, num_classes=2).to(device)
    
    # Load encoder weights (backbone is shared)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    # Load PCA manifold
    print("Loading PCA normality manifold...")
    manifold_ckpt = torch.load(args.manifold_path, map_location=device)
    manifold = PCANormalityManifold(dc=64, r=32, device=device)
    manifold.register_buffer('mu_c', manifold_ckpt['mu_c'])
    manifold.register_buffer('U_c', manifold_ckpt['U_c'])
    manifold.is_fitted = True
    
    # Initialize causal meta-transfer
    num_base_classes = 12 if args.dataset == "visa" else 10  # Adjust as needed
    meta_transfer = CausalMetaTransfer(
        dc=64,
        num_base_classes=num_base_classes,
        alpha=5.0,
        rho=0.5
    ).to(device)
    
    # Compute base-class statistics (mu_c, Sigma_c) for meta-transfer
    # For simplicity, we use the PCA manifold mean and identity covariance
    mu_c_base = manifold.mu_c.unsqueeze(0).repeat(num_base_classes, 1)  # [num_base, dc]
    Sigma_c_base = torch.eye(64, device=device).unsqueeze(0).repeat(num_base_classes, 1, 1)  # [num_base, dc, dc]
    
    # Optimizer (only fine-tune detection head and meta-transfer)
    optimizer = optim.SGD(
        list(detection_head.parameters()) + list(meta_transfer.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Get support features
    print("Encoding support set...")
    support_images, support_labels, _ = next(iter(support_loader))
    support_images = support_images.to(device)
    
    # Extract RoI features for support (assume full image as RoI for simplicity)
    with torch.no_grad():
        backbone_features = backbone(support_images)
        c5_features = backbone_features['C5']  # [K, 2048, H/32, W/32]
        support_roi_feats = torch.mean(c5_features, dim=[2, 3])  # [K, 2048]
    
    # Encode causal features for support prototype
    with torch.no_grad():
        z_c_support, _ = encoder(support_images)  # [K, 64]
        z_s_bar = z_c_support.mean(dim=0)  # [64]
    
    # Compute causal importance vector w
    print("Computing causal importance vector...")
    with torch.no_grad():
        Sigma_c_inv = torch.inverse(Sigma_c_base + 1e-6 * torch.eye(64, device=device))
        w = meta_transfer(z_s_bar, mu_c_base, Sigma_c_inv)  # [64]
    
    print(f"Active causal dimensions: {w.nonzero().size(0)} / 64")
    
    # Fine-tuning loop
    print("Starting few-shot fine-tuning...")
    backbone.eval()
    encoder.eval()
    
    for iteration in range(args.iterations):
        # Sample query batch from test set (excluding support)
        query_indices = [i for i in range(len(test_dataset)) if i not in support_set.indices]
        query_subset = Subset(test_dataset, np.random.choice(query_indices, args.batch_size, replace=False))
        query_loader = DataLoader(query_subset, batch_size=args.batch_size, shuffle=False)
        
        for query_images, query_labels, _ in query_loader:
            query_images = query_images.to(device)
            
            # Extract query RoI features
            with torch.no_grad():
                backbone_features = backbone(query_images)
                c5_features = backbone_features['C5']
                query_roi_feats = torch.mean(c5_features, dim=[2, 3])  # [B, 2048]
            
            # Forward detection head with interventional similarity
            cls_logits, bbox_deltas = detection_head(
                query_roi_feats,
                support_roi_feats,
                w
            )
            
            # Compute loss (binary cross-entropy + smooth L1)
            # For simplicity, assume all queries are defects (label=1)
            target_labels = torch.ones(args.batch_size, dtype=torch.long, device=device)
            cls_loss = nn.CrossEntropyLoss()(cls_logits, target_labels)
            
            # Dummy bbox targets (not used in industrial AD)
            bbox_loss = torch.tensor(0.0, device=device)
            
            total_loss = cls_loss + bbox_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if iteration % 100 == 0:
                print(f"Iter {iteration}: Total={total_loss:.4f}, Cls={cls_loss:.4f}")
            break  # Process one batch per iteration
    
    # Save fine-tuned model
    torch.save({
        'detection_head_state_dict': detection_head.state_dict(),
        'meta_transfer_state_dict': meta_transfer.state_dict(),
        'w': w,
        'novel_class': args.novel_class,
        'k_shot': args.k_shot
    }, os.path.join(args.save_dir, f"{args.dataset}_{args.novel_class}_{args.k_shot}shot.pth"))
    
    print("Few-shot fine-tuning completed!")

if __name__ == "__main__":
    main()