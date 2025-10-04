import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from perlin_noise import PerlinNoise

def perlin_noise_2d(shape, frequency=8, persistence=0.5, seed=None):
    """Generate 2D Perlin noise."""
    noise = PerlinNoise(octaves=frequency, seed=seed)
    h, w = shape
    scale = 1.0 / frequency
    noise_map = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            noise_map[i][j] = noise([i * scale, j * scale])
    # Normalize to [0, 1]
    noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    return noise_map

def apply_surface_corruption(image, alpha_s=0.3, beta_s=0.02, omega=8, psi=0.5, seed=None):
    """
    Simulate surface defects (scratches, dents, cracks) using Perlin noise.
    
    Args:
        image (np.ndarray): Input image (H, W, 3), uint8.
        alpha_s (float): Defect intensity (0.0–1.0).
        beta_s (float): Sensor noise level.
        omega (int): Perlin noise frequency.
        psi (float): Persistence (not directly used; octaves controlled by omega).
        seed (int, optional): Random seed.
    
    Returns:
        np.ndarray: Corrupted image (H, W, 3), uint8.
    """
    H, W = image.shape[:2]
    perlin = perlin_noise_2d((H, W), frequency=omega, persistence=psi, seed=seed)
    # Create defect mask: darker regions = defects
    defect_mask = 1.0 - perlin  # Invert so noise valleys = scratches
    defect_mask = np.clip(defect_mask, 0, 1)
    
    # Apply defect: darken image where defect_mask is high
    image_float = image.astype(np.float32) / 255.0
    corrupted = image_float * (1.0 - alpha_s * defect_mask[..., None])
    
    # Add sensor noise
    noise = np.random.normal(0, beta_s, corrupted.shape)
    corrupted = np.clip(corrupted + noise, 0, 1)
    
    return (corrupted * 255).astype(np.uint8)

def apply_structural_corruption(image, mask=None, p_drop=0.2, shift_range=32, seed=None):
    """
    Emulate structural defects (missing/misplaced components) via masking and shifting.
    
    Args:
        image (np.ndarray): Input image (H, W, 3), uint8.
        mask (np.ndarray, optional): Binary mask (H, W), uint8. If None, use whole image.
        p_drop (float): Probability to drop a component (0.0–1.0).
        shift_range (int): Max shift in pixels (±shift_range).
        seed (int, optional): Random seed.
    
    Returns:
        np.ndarray: Corrupted image (H, W, 3), uint8.
    """
    if seed is not None:
        np.random.seed(seed)
    
    H, W = image.shape[:2]
    image_corrupted = image.copy()
    
    if mask is None:
        mask = np.ones((H, W), dtype=np.uint8)
    
    # Randomly drop components (set to mean background)
    if np.random.rand() < p_drop:
        bg_color = image[mask == 0].mean(axis=0) if mask is not None else image.mean(axis=(0,1))
        image_corrupted[mask == 1] = bg_color.astype(np.uint8)
    
    # Randomly shift components
    dx = np.random.randint(-shift_range, shift_range + 1)
    dy = np.random.randint(-shift_range, shift_range + 1)
    
    # Create translation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(image_corrupted, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    
    # Preserve background where mask=0
    if mask is not None:
        shifted[mask == 0] = image[mask == 0]
    
    return shifted

def apply_contamination_corruption(image, n_blobs=3, gamma_k=0.4, mu_c=[0.8, 0.2, 0.2], seed=None):
    """
    Inject color spots/specks via Gaussian blob superposition.
    
    Args:
        image (np.ndarray): Input image (H, W, 3), uint8.
        n_blobs (int): Number of blobs to add.
        gamma_k (float): Intensity of each blob (0.0–1.0).
        mu_c (list): Mean color of blobs in RGB [0,1] (e.g., red spot).
        seed (int, optional): Random seed.
    
    Returns:
        np.ndarray: Corrupted image (H, W, 3), uint8.
    """
    if seed is not None:
        np.random.seed(seed)
    
    H, W = image.shape[:2]
    image_float = image.astype(np.float32) / 255.0
    corrupted = image_float.copy()
    
    for _ in range(n_blobs):
        # Random center
        cx = np.random.randint(0, W)
        cy = np.random.randint(0, H)
        # Random sigma
        sigma = np.random.uniform(5, 15)
        # Create 2D Gaussian blob
        x = np.arange(W)
        y = np.arange(H)
        X, Y = np.meshgrid(x, y)
        blob = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        blob = blob / blob.max()  # Normalize to [0,1]
        # Color vector
        color = np.random.normal(mu_c, 0.1, size=3)
        color = np.clip(color, 0, 1)
        # Add blob
        for c in range(3):
            corrupted[:, :, c] += gamma_k * blob * color[c]
    
    corrupted = np.clip(corrupted, 0, 1)
    return (corrupted * 255).astype(np.uint8)

def apply_corruption(image, task_type="surface", **kwargs):
    """
    Apply corruption based on task type.
    
    Args:
        image (np.ndarray): Input image (H, W, 3), uint8.
        task_type (str): One of {"surface", "structural", "contamination"}.
        **kwargs: Corruption-specific parameters.
    
    Returns:
        np.ndarray: Corrupted image.
    """
    if task_type == "surface":
        return apply_surface_corruption(image, **kwargs)
    elif task_type == "structural":
        return apply_structural_corruption(image, **kwargs)
    elif task_type == "contamination":
        return apply_contamination_corruption(image, **kwargs)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")