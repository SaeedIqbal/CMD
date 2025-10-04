import os
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class IndustrialDatasetBase(Dataset):
    """Base class for industrial anomaly detection datasets."""
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)


class VisaDataset(IndustrialDatasetBase):
    """VisA: Visual Anomaly Detection Benchmark."""
    def _load_samples(self):
        # VisA structure: root/category/split/good/ or split/anomaly_type/
        categories = [d for d in self.root.iterdir() if d.is_dir()]
        for cat in categories:
            if self.split == "train":
                # Only 'good' images for training
                img_dir = cat / "train" / "good"
                if img_dir.exists():
                    for img in img_dir.glob("*.jpg"):
                        self.samples.append((img, 0))
            else:
                # Test: 'good' (label=0) + anomalies (label=1)
                for split_dir in ["test"]:
                    good_dir = cat / split_dir / "good"
                    if good_dir.exists():
                        for img in good_dir.glob("*.jpg"):
                            self.samples.append((img, 0))
                    anomaly_types = [d for d in (cat / split_dir).iterdir() if d.is_dir() and d.name != "good"]
                    for anomaly_dir in anomaly_types:
                        for img in anomaly_dir.glob("*.jpg"):
                            self.samples.append((img, 1))


class MVTecADDataset(IndustrialDatasetBase):
    """MVTec Anomaly Detection Dataset."""
    def _load_samples(self):
        # MVTec-AD: root/category/train/good/, test/good/ and test/anomaly_type/
        categories = [d for d in self.root.iterdir() if d.is_dir()]
        for cat in categories:
            if self.split == "train":
                img_dir = cat / "train" / "good"
                if img_dir.exists():
                    for img in img_dir.glob("*.png"):
                        self.samples.append((img, 0))
            else:
                # Test split
                good_dir = cat / "test" / "good"
                if good_dir.exists():
                    for img in good_dir.glob("*.png"):
                        self.samples.append((img, 0))
                anomaly_types = [d for d in (cat / "test").iterdir() if d.is_dir() and d.name != "good"]
                for anomaly_dir in anomaly_types:
                    for img in anomaly_dir.glob("*.png"):
                        self.samples.append((img, 1))


class MVTecLOCODataset(IndustrialDatasetBase):
    """MVTec LOCO AD: Logical and Structural Anomalies."""
    def _load_samples(self):
        # MVTec-LOCO: train/good, test/good, test/logical_anomalies, test/structural_anomalies
        categories = [d for d in self.root.iterdir() if d.is_dir()]
        for cat in categories:
            if self.split in ["train", "val"]:
                img_dir = cat / self.split / "good"
                if img_dir.exists():
                    for img in img_dir.glob("*.png"):
                        self.samples.append((img, 0))
            else:  # test
                for subdir in ["good", "logical_anomalies", "structural_anomalies"]:
                    img_dir = cat / "test" / subdir
                    if img_dir.exists():
                        label = 0 if subdir == "good" else 1
                        for img in img_dir.glob("*.png"):
                            self.samples.append((img, label))


class RealIADDataset(IndustrialDatasetBase):
    """Real-IAD: Real-World Multi-View Industrial Anomaly Detection."""
    def _load_samples(self):
        # Real-IAD: root/code/ (e.g., AK/, BX/, etc.), each contains good/ and ko/
        defect_codes = [d for d in self.root.iterdir() if d.is_dir()]
        for code_dir in defect_codes:
            if self.split == "train":
                good_dir = code_dir / "good"
                if good_dir.exists():
                    for img in good_dir.glob("*.png"):
                        self.samples.append((img, 0))
            else:
                # Test: good (0) and ko (1)
                good_dir = code_dir / "good"
                ko_dir = code_dir / "ko"
                if good_dir.exists():
                    for img in good_dir.glob("*.png"):
                        self.samples.append((img, 0))
                if ko_dir.exists():
                    for img in ko_dir.glob("*.png"):
                        self.samples.append((img, 1))


class BTADDataset(IndustrialDatasetBase):
    """BTAD: beanTech Anomaly Detection."""
    def _load_samples(self):
        # BTAD: root/product_X/train/ (ok), root/product_X/test/ (ok + ko)
        products = [d for d in self.root.iterdir() if d.is_dir() and d.name.startswith("product_")]
        for prod in products:
            if self.split == "train":
                img_dir = prod / "train" / "ok"
                if img_dir.exists():
                    for img in img_dir.glob("*.bmp"):
                        self.samples.append((img, 0))
            else:
                # Test: ok (0) and ko (1)
                ok_dir = prod / "test" / "ok"
                ko_dir = prod / "test" / "ko"
                if ok_dir.exists():
                    for img in ok_dir.glob("*.bmp"):
                        self.samples.append((img, 0))
                if ko_dir.exists():
                    for img in ko_dir.glob("*.bmp"):
                        self.samples.append((img, 1))


class DGADDataset(IndustrialDatasetBase):
    """DGAD: Difficult Generalist Anomaly Detection."""
    def _load_samples(self):
        # DGAD: root/class/train/good/, test/good/ and test/anomaly_type/
        classes = [d for d in self.root.iterdir() if d.is_dir()]
        for cls in classes:
            if self.split == "train":
                img_dir = cls / "train" / "good"
                if img_dir.exists():
                    for img in img_dir.glob("*.png"):
                        self.samples.append((img, 0))
            else:
                good_dir = cls / "test" / "good"
                if good_dir.exists():
                    for img in good_dir.glob("*.png"):
                        self.samples.append((img, 0))
                anomaly_types = [d for d in (cls / "test").iterdir() if d.is_dir() and d.name != "good"]
                for anomaly_dir in anomaly_types:
                    for img in anomaly_dir.glob("*.png"):
                        self.samples.append((img, 1))


class MPDDDataset(IndustrialDatasetBase):
    """MPDD: Metal Parts Defect Detection Dataset."""
    def _load_samples(self):
        # MPDD: root/class/train/good/, test/good/ and test/defect_type/
        classes = [d for d in self.root.iterdir() if d.is_dir()]
        for cls in classes:
            if self.split == "train":
                img_dir = cls / "train" / "good"
                if img_dir.exists():
                    for img in img_dir.glob("*.png"):
                        self.samples.append((img, 0))
            else:
                good_dir = cls / "test" / "good"
                if good_dir.exists():
                    for img in good_dir.glob("*.png"):
                        self.samples.append((img, 0))
                defect_types = [d for d in (cls / "test").iterdir() if d.is_dir() and d.name != "good"]
                for defect_dir in defect_types:
                    for img in defect_dir.glob("*.png"):
                        self.samples.append((img, 1))