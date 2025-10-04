import os
from .industrial_dataset import (
    VisaDataset,
    MVTecADDataset,
    MVTecLOCODataset,
    RealIADDataset,
    BTADDataset,
    DGADDataset,
    MPDDDataset
)

# Root path for all datasets
DATASET_ROOT = "/home/phd/datasets/"

# Mapping from dataset name to class and root path
DATASET_CONFIG = {
    "visa": {
        "class": VisaDataset,
        "root": os.path.join(DATASET_ROOT, "VisA")
    },
    "mvtec_ad": {
        "class": MVTecADDataset,
        "root": os.path.join(DATASET_ROOT, "MVTec-AD")
    },
    "mvtec_loco": {
        "class": MVTecLOCODataset,
        "root": os.path.join(DATASET_ROOT, "MVTec-LOCO")
    },
    "real_iad": {
        "class": RealIADDataset,
        "root": os.path.join(DATASET_ROOT, "Real-IAD")
    },
    "btad": {
        "class": BTADDataset,
        "root": os.path.join(DATASET_ROOT, "BTAD")
    },
    "dgad": {
        "class": DGADDataset,
        "root": os.path.join(DATASET_ROOT, "DGAD")
    },
    "mpdd": {
        "class": MPDDDataset,
        "root": os.path.join(DATASET_ROOT, "MPDD")
    }
}

def get_dataset(name, split="train", transform=None, **kwargs):
    """
    Factory function to load industrial anomaly detection datasets.

    Args:
        name (str): Dataset name (e.g., 'visa', 'mvtec_ad', 'real_iad', etc.)
        split (str): Split to load ('train', 'val', 'test')
        transform (callable, optional): Transform to apply to images
        **kwargs: Additional arguments passed to dataset class

    Returns:
        torch.utils.data.Dataset: The dataset instance
    """
    if name not in DATASET_CONFIG:
        raise ValueError(f"Dataset '{name}' not supported. Choose from {list(DATASET_CONFIG.keys())}")

    config = DATASET_CONFIG[name]
    dataset_class = config["class"]
    root = config["root"]

    if not os.path.exists(root):
        raise FileNotFoundError(
            f"Dataset root not found: {root}. "
            f"Please download the dataset and place it at {DATASET_ROOT}{name.upper()}"
        )

    return dataset_class(root=root, split=split, transform=transform, **kwargs)