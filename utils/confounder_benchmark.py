import numpy as np
import cv2
from PIL import Image, ImageEnhance
import os
from pathlib import Path

class ConfounderBenchmark:
    """
    Synthetic confounder benchmark for robustness evaluation.
    
    Applies class-specific distractors to non-defective regions:
        - Glare: Phong shading model for specular highlights
        - Texture overlays: Superimpose background textures
        - Pose jitter: Random rotation/translation
    
    Args:
        dataset_name (str): Name of dataset (e.g., 'visa', 'mpdd', 'real_iad')
        data_root (str): Root directory of dataset
    """
    def __init__(self, dataset_name, data_root):
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self._load_texture_bank()
    
    def _load_texture_bank(self):
        """Load background textures for overlay (from DGAD or dataset-specific sources)."""
        if self.dataset_name == 'dgad':
            texture_dir = self.data_root / "train" / "good"
        else:
            # Use DGAD textures as universal bank (place DGAD at /home/phd/datasets/DGAD)
            dgad_path = Path("/home/phd/datasets/DGAD")
            if dgad_path.exists():
                texture_dir = dgad_path / "train" / "good"
            else:
                texture_dir = None
        
        self.textures = []
        if texture_dir and texture_dir.exists():
            for img_path in texture_dir.glob("*.png"):
                try:
                    texture = np.array(Image.open(img_path).convert("RGB"))
                    self.textures.append(texture)
                except:
                    continue
        if not self.textures:
            # Fallback: generate procedural noise
            self.textures = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(10)]
    
    def apply_glare(self, image, intensity=0.7, center=None, size=50):
        """
        Apply specular glare using Phong shading model.
        
        Args:
            image (np.ndarray): [H, W, 3] input image
            intensity (float): Glare intensity (0.0–1.0)
            center (tuple, optional): (x, y) center of glare
            size (int): Radius of glare effect
        
        Returns:
            np.ndarray: Image with glare
        """
        H, W = image.shape[:2]
        if center is None:
            center = (np.random.randint(W//4, 3*W//4), np.random.randint(H//4, 3*H//4))
        
        # Create glare mask
        x = np.arange(W)
        y = np.arange(H)
        X, Y = np.meshgrid(x, y)
        dist_sq = (X - center[0])**2 + (Y - center[1])**2
        glare_mask = np.exp(-dist_sq / (2 * (size**2)))
        glare_mask = np.clip(glare_mask * intensity, 0, 1)
        
        # Apply glare (additive in RGB)
        glare = np.ones_like(image) * 255  # White highlight
        corrupted = image.astype(np.float32) * (1 - glare_mask[..., None]) + glare * glare_mask[..., None]
        return np.clip(corrupted, 0, 255).astype(np.uint8)
    
    def apply_texture_overlay(self, image, alpha=0.3):
        """
        Overlay background texture on non-defective regions.
        
        Args:
            image (np.ndarray): [H, W, 3] input image
            alpha (float): Overlay opacity (0.0–1.0)
        
        Returns:
            np.ndarray: Image with texture overlay
        """
        H, W = image.shape[:2]
        texture = self.textures[np.random.randint(len(self.textures))]
        
        # Resize texture to match image
        texture = cv2.resize(texture, (W, H))
        
        # Blend
        corrupted = cv2.addWeighted(image, 1 - alpha, texture, alpha, 0)
        return corrupted.astype(np.uint8)
    
    def apply_pose_jitter(self, image, max_angle=15, max_shift=32):
        """
        Apply random rotation and translation.
        
        Args:
            image (np.ndarray): [H, W, 3] input image
            max_angle (float): Max rotation angle (degrees)
            max_shift (int): Max translation (pixels)
        
        Returns:
            np.ndarray: Transformed image
        """
        H, W = image.shape[:2]
        
        # Random rotation
        angle = np.random.uniform(-max_angle, max_angle)
        scale = 1.0
        center = (W // 2, H // 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Random translation
        tx = np.random.randint(-max_shift, max_shift + 1)
        ty = np.random.randint(-max_shift, max_shift + 1)
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty
        
        # Apply transformation
        corrupted = cv2.warpAffine(image, M_rot, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return corrupted
    
    def corrupt_image(self, image, confounder_type, **kwargs):
        """
        Apply confounder based on dataset and defect type.
        
        Args:
            image (np.ndarray): [H, W, 3] input image
            confounder_type (str): One of {'glare', 'texture', 'pose'}
            **kwargs: Confounder-specific parameters
        
        Returns:
            np.ndarray: Corrupted image
        """
        if confounder_type == "glare":
            return self.apply_glare(image, **kwargs)
        elif confounder_type == "texture":
            return self.apply_texture_overlay(image, **kwargs)
        elif confounder_type == "pose":
            return self.apply_pose_jitter(image, **kwargs)
        else:
            raise ValueError(f"Unknown confounder_type: {confounder_type}")
    
    def get_confounder_type(self, dataset_name, category=None):
        """
        Determine confounder type based on dataset and category.
        
        Args:
            dataset_name (str): Dataset name
            category (str, optional): Object category (e.g., 'pcb', 'capsules')
        
        Returns:
            str: Confounder type
        """
        if dataset_name in ['mpdd', 'real_iad']:
            return "glare"
        elif dataset_name in ['visa', 'btad']:
            return "texture"
        elif dataset_name == 'mvtec_loco':
            return "pose"
        else:
            # Default fallback
            return "glare"
    
    def evaluate_robustness_gap(self, model, clean_loader, confounded_loader):
        """
        Compute robustness gap: Δ_conf = nAP50_clean - nAP50_confounded.
        
        Args:
            model: CMD model with detect() method
            clean_loader: DataLoader for clean images
            confounded_loader: DataLoader for confounded images
        
        Returns:
            dict: {
                'nap50_clean': float,
                'nap50_confounded': float,
                'robustness_gap': float
            }
        """
        from utils.metrics import compute_metrics_torch
        
        # Evaluate on clean data
        clean_preds = []
        clean_gts = []
        for images, targets in clean_loader:
            preds = model.detect(images)
            clean_preds.extend(preds)
            clean_gts.extend(targets)
        
        clean_metrics = compute_metrics_torch(
            [p['boxes'] for p in clean_preds],
            [p['scores'] for p in clean_preds],
            [p['labels'] for p in clean_preds],
            [t['boxes'] for t in clean_gts],
            [t['labels'] for t in clean_gts]
        )
        
        # Evaluate on confounded data
        confounded_preds = []
        confounded_gts = []
        for images, targets in confounded_loader:
            preds = model.detect(images)
            confounded_preds.extend(preds)
            confounded_gts.extend(targets)
        
        confounded_metrics = compute_metrics_torch(
            [p['boxes'] for p in confounded_preds],
            [p['scores'] for p in confounded_preds],
            [p['labels'] for p in confounded_preds],
            [t['boxes'] for t in confounded_gts],
            [t['labels'] for t in confounded_gts]
        )
        
        return {
            'nap50_clean': clean_metrics['nap50'],
            'nap50_confounded': confounded_metrics['nap50'],
            'robustness_gap': clean_metrics['nap50'] - confounded_metrics['nap50']
        }