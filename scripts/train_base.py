import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.causal_encoder import CausalEncoder
from models.decoder import UNetDecoder
from losses.vae_loss import vae_loss, reparameterize
from losses.hsic import HSICLoss
from losses.ortho_loss import OrthoLoss
from losses.normality_deviation import NormalityDeviationLoss
from data.dataset_factory import get_dataset
from data.corruption import apply_corruption
from utils.pca_normality import PCANormalityManifold
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="CMD Base Meta-Pretraining")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="mvtec_ad", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=20000, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.006, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Save directory")
    return parser.parse_args()

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
    
    # Load normal dataset (train split only)
    dataset = get_dataset(args.dataset, split="train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Initialize models
    encoder = CausalEncoder(dc=64, dn=192, pretrained=True).to(device)
    decoder = UNetDecoder(dc=64, dn=192).to(device)
    
    # Initialize losses
    hsic_loss_fn = HSICLoss()
    ortho_loss_fn = OrthoLoss()
    nd_loss_fn = NormalityDeviationLoss(
        lambda_hsic=1.2,
        mu_smooth=0.05,
        r_pca=32,
        device=device
    )
    
    # Optimizer
    optimizer = optim.SGD(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Collect normal latents for PCA manifold
    normal_latents = []
    
    print("Stage 1: Collecting normal latents for PCA manifold...")
    encoder.eval()
    with torch.no_grad():
        for images, _, _ in dataloader:
            images = images.to(device)
            z_c, z_n = encoder(images)
            normal_latents.append(z_c.cpu())
    
    normal_latents = torch.cat(normal_latents, dim=0)
    print(f"Collected {normal_latents.size(0)} normal latents")
    
    # Fit normality manifold
    print("Fitting PCA normality manifold...")
    explained_ratio = nd_loss_fn.fit_normality_manifold(normal_latents)
    print(f"Explained variance (top 32): {explained_ratio.sum():.2%}")
    
    # Save manifold
    torch.save({
        'mu_c': nd_loss_fn.mu_c,
        'U_c': nd_loss_fn.U_c,
        'explained_ratio': explained_ratio
    }, os.path.join(args.save_dir, f"{args.dataset}_manifold.pth"))
    
    # Training loop
    print("Stage 2: Meta-pretraining with corruption tasks...")
    encoder.train()
    decoder.train()
    
    task_types = ["surface", "structural", "contamination"]
    corruption_params = {
        "surface": {"alpha_s": 0.3, "beta_s": 0.02, "omega": 8, "psi": 0.5},
        "structural": {"p_drop": 0.2, "shift_range": 32},
        "contamination": {"n_blobs": 3, "gamma_k": 0.4, "mu_c": [0.8, 0.2, 0.2]}
    }
    
    iteration = 0
    for epoch in range(args.epochs // len(dataloader) + 1):
        for images, _, img_paths in dataloader:
            if iteration >= args.epochs:
                break
                
            # Sample corruption task
            task_type = np.random.choice(task_types)
            params = corruption_params[task_type]
            
            # Apply corruption to normal images
            corrupted_images = []
            for img, path in zip(images, img_paths):
                # Convert to numpy (HWC, uint8)
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                corrupted_np = apply_corruption(img_np, task_type=task_type, **params)
                # Convert back to tensor
                corrupted_tensor = torch.from_numpy(corrupted_np).permute(2, 0, 1).float() / 255.0
                corrupted_images.append(corrupted_tensor)
            
            corrupted_batch = torch.stack(corrupted_images).to(device)
            
            # Forward pass
            z_c, z_n = encoder(corrupted_batch)
            
            # VAE reparameterization (optional)
            # For simplicity, we use deterministic encoding
            recon = decoder(z_c, z_n)
            
            # Losses
            vae_losses = vae_loss(
                recon,
                corrupted_batch,
                z_c,  # mu_c
                torch.zeros_like(z_c),  # logvar_c (deterministic)
                z_n,  # mu_n
                torch.zeros_like(z_n),  # logvar_n
                beta_kl=0.5
            )
            
            hsic = hsic_loss_fn(z_c, z_n)
            ortho = ortho_loss_fn(z_c, z_n)
            
            # Normality deviation loss (requires feature maps for smoothness)
            # For simplicity, we skip smoothness here or use z_c directly
            nd_losses = {
                'deviation': nd_loss_fn.deviation(z_c).mean(),
                'hsic': hsic,
                'smooth': torch.tensor(0.0, device=device)
            }
            nd_total = nd_losses['deviation'] + 1.2 * nd_losses['hsic'] + 0.05 * nd_losses['smooth']
            
            total_loss = vae_losses['total'] + 1.0 * ortho + nd_total
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if iteration % 100 == 0:
                print(f"Iter {iteration}: Total={total_loss:.4f}, Recon={vae_losses['recon']:.4f}, "
                      f"Dev={nd_losses['deviation']:.4f}, HSIC={hsic:.4f}, Ortho={ortho:.4f}")
            
            iteration += 1
    
    # Save final model
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'manifold_path': os.path.join(args.save_dir, f"{args.dataset}_manifold.pth")
    }, os.path.join(args.save_dir, f"{args.dataset}_base_model.pth"))
    
    print("Base meta-pretraining completed!")

if __name__ == "__main__":
    main()