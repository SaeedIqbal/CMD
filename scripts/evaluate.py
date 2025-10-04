import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from models.causal_encoder import CausalEncoder
from models.meta_transfer import CausalMetaTransfer
from models.detection_head import InterventionalDetectionHead
from models.backbone import build_detection_backbone
from data.dataset_factory import get_dataset
from utils.metrics import compute_metrics_torch
from utils.confounder_benchmark import ConfounderBenchmark
from utils.pca_normality import PCANormalityManifold
import json

def parse_args():
    parser = argparse.ArgumentParser(description="CMD Evaluation")
    parser.add_argument("--config", type=str, default="configs/eval.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--k_shot", type=int, default=3, choices=[1, 2, 3, 5, 10], help="Number of support shots")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--base_ckpt", type=str, required=True, help="Path to base model checkpoint")
    parser.add_argument("--manifold_path", type=str, required=True, help="Path to PCA manifold")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    return parser.parse_args()

def load_few_shot_model(args, device):
    """Load fine-tuned CMD model for evaluation."""
    # Load base encoder
    encoder = CausalEncoder(dc=64, dn=192, pretrained=False).to(device)
    base_ckpt = torch.load(args.base_ckpt, map_location=device)
    encoder.load_state_dict(base_ckpt['encoder_state_dict'])
    
    # Load detection head and meta-transfer
    detection_head = InterventionalDetectionHead(in_channels=2048, hidden_dim=1024, num_classes=2).to(device)
    meta_transfer = CausalMetaTransfer(dc=64, num_base_classes=12 if args.dataset == "visa" else 10).to(device)
    
    # Load fine-tuned weights
    ckpt = torch.load(args.checkpoint, map_location=device)
    detection_head.load_state_dict(ckpt['detection_head_state_dict'])
    meta_transfer.load_state_dict(ckpt['meta_transfer_state_dict'])
    w = ckpt['w'].to(device)
    
    return encoder, detection_head, meta_transfer, w

def evaluate_clean(model_components, test_loader, device):
    """Evaluate on clean test set."""
    encoder, detection_head, meta_transfer, w = model_components
    backbone = build_detection_backbone(pretrained=False).to(device)
    backbone.eval()
    encoder.eval()
    detection_head.eval()
    
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            
            # Extract features
            backbone_features = backbone(images)
            c5_features = backbone_features['C5']
            query_roi_feats = torch.mean(c5_features, dim=[2, 3])  # [B, 2048]
            
            # Dummy support features (use query as support for simplicity)
            support_roi_feats = query_roi_feats.clone()
            
            # Detect
            cls_logits, bbox_deltas = detection_head(query_roi_feats, support_roi_feats, w)
            
            # Format predictions
            scores = torch.softmax(cls_logits, dim=1)[:, 1]  # defect scores
            labels = (scores > 0.5).long()
            boxes = torch.zeros(scores.size(0), 4).to(device)  # dummy boxes
            
            preds = {
                'boxes': boxes.cpu(),
                'scores': scores.cpu(),
                'labels': labels.cpu()
            }
            gts = {
                'boxes': targets['boxes'],
                'labels': targets['labels']
            }
            
            all_preds.append(preds)
            all_gts.append(gts)
    
    return all_preds, all_gts

def evaluate_confounded(model_components, test_loader, dataset_name, device):
    """Evaluate on confounded test set."""
    benchmark = ConfounderBenchmark(dataset_name, f"/home/phd/datasets/{dataset_name.upper()}")
    
    encoder, detection_head, meta_transfer, w = model_components
    backbone = build_detection_backbone(pretrained=False).to(device)
    backbone.eval()
    encoder.eval()
    detection_head.eval()
    
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for images, targets, img_paths in test_loader:
            # Apply confounders
            confounded_images = []
            for img, path in zip(images, img_paths):
                img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                conf_type = benchmark.get_confounder_type(dataset_name)
                corrupted_np = benchmark.corrupt_image(img_np, conf_type)
                corrupted_tensor = torch.from_numpy(corrupted_np).permute(2, 0, 1).float() / 255.0
                confounded_images.append(corrupted_tensor)
            
            confounded_batch = torch.stack(confounded_images).to(device)
            
            # Extract features
            backbone_features = backbone(confounded_batch)
            c5_features = backbone_features['C5']
            query_roi_feats = torch.mean(c5_features, dim=[2, 3])
            
            # Dummy support features
            support_roi_feats = query_roi_feats.clone()
            
            # Detect
            cls_logits, bbox_deltas = detection_head(query_roi_feats, support_roi_feats, w)
            
            # Format predictions
            scores = torch.softmax(cls_logits, dim=1)[:, 1]
            labels = (scores > 0.5).long()
            boxes = torch.zeros(scores.size(0), 4).to(device)
            
            preds = {
                'boxes': boxes.cpu(),
                'scores': scores.cpu(),
                'labels': labels.cpu()
            }
            gts = {
                'boxes': targets['boxes'],
                'labels': targets['labels']
            }
            
            all_preds.append(preds)
            all_gts.append(gts)
    
    return all_preds, all_gts

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading CMD model...")
    model_components = load_few_shot_model(args, device)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = get_dataset(args.dataset, split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4)
    
    # Evaluate on clean data
    print("Evaluating on clean test set...")
    clean_preds, clean_gts = evaluate_clean(model_components, test_loader, device)
    clean_metrics = compute_metrics_torch(
        [p['boxes'] for p in clean_preds],
        [p['scores'] for p in clean_preds],
        [p['labels'] for p in clean_preds],
        [t['boxes'] for t in clean_gts],
        [t['labels'] for t in clean_gts]
    )
    
    # Evaluate on confounded data
    print("Evaluating on confounded test set...")
    confounded_preds, confounded_gts = evaluate_confounded(model_components, test_loader, args.dataset, device)
    confounded_metrics = compute_metrics_torch(
        [p['boxes'] for p in confounded_preds],
        [p['scores'] for p in confounded_preds],
        [p['labels'] for p in confounded_preds],
        [t['boxes'] for t in confounded_gts],
        [t['labels'] for t in confounded_gts]
    )
    
    # Compute robustness gap
    robustness_gap = clean_metrics['nap50'] - confounded_metrics['nap50']
    
    # Save results
    results = {
        'dataset': args.dataset,
        'k_shot': args.k_shot,
        'clean_metrics': {k: float(v) for k, v in clean_metrics.items()},
        'confounded_metrics': {k: float(v) for k, v in confounded_metrics.items()},
        'robustness_gap': float(robustness_gap)
    }
    
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{args.k_shot}shot_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation Results for {args.dataset} ({args.k_shot}-shot):")
    print(f"  nAP50 (clean):     {clean_metrics['nap50']:.2f}")
    print(f"  nAP50 (confounded):{confounded_metrics['nap50']:.2f}")
    print(f"  Robustness Gap:    {robustness_gap:.2f}")
    if 'pixel_auroc' in clean_metrics:
        print(f"  Pixel AUROC:       {clean_metrics['pixel_auroc']:.2f}")
    if 'f1_score' in clean_metrics:
        print(f"  Localization F1:   {clean_metrics['f1_score']:.2f}")
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()