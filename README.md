
# Causal Meta-Disentanglement (CMD) for Robust Few-Shot Object Detection

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

**Official PyTorch implementation** of **"Causal Meta-Disentanglement for Robust Few-Shot Object Detection"**, a novel framework that redefines industrial few-shot object detection as **causal deviation detection**, not class similarity.

> **Key Insight**: Traditional FSOD methods (e.g., DLFG) fail in industrial settings because they conflate **causal defect signals** (e.g., micro-cracks) with **non-causal confounders** (e.g., glare, pose). CMD enforces **causal identifiability** in latent representations, achieving state-of-the-art robustness.

---

## 📌 Highlights

- ✅ **First FSOD framework with causal disentanglement**: Explicitly separates \( \mathcal{Z}_c \) (defect generators) from \( \mathcal{Z}_n \) (nuisances).
- ✅ **Meta-pretraining via normality deviation**: Learns defect semantics from **defect-free base data** using procedural corruption.
- ✅ **Interventional similarity \( \mathcal{S}_{\text{do}} \)**: Computes similarity only over causally relevant features, ensuring invariance to confounders.
- ✅ **+4.2% avg nAP50 gain** over SOTA across **7 industrial datasets**.
- ✅ **Robustness gap reduced by 67%** under synthetic confounders (glare, texture, pose).

---

## 📊 Results

### Main Results (3-shot, avg over 7 datasets)
| Method       | nAP50 (%) ↑ | Pixel AUROC (%) ↑ | F1-Score (%) ↑ | Robustness Gap ↓ |
|--------------|-------------|-------------------|----------------|------------------|
| TFA          | 40.7        | 84.2              | 70.9           | 6.9              |
| Meta R-CNN   | 43.0        | 86.1              | 73.2           | 6.5              |
| DeFRCN       | 45.5        | 87.2              | 74.0           | 6.2              |
| FSCE         | 46.9        | 87.6              | 74.3           | 6.2              |
| **DLFG**     | **49.2**    | **87.9**          | **74.7**       | **6.2**          |
| **CMD (Ours)** | **53.4**    | **92.9**          | **81.5**       | **2.3**          |

> **CMD outperforms DLFG by +4.2% nAP50** and reduces robustness gap from **6.2 → 2.3**.

### Per-Dataset nAP50 (3-shot)
| Dataset       | DLFG   | CMD    | Gain   |
|---------------|--------|--------|--------|
| VisA          | 50.4   | 55.1   | +4.7   |
| MVTec-AD      | 52.9   | 57.6   | +4.7   |
| MVTec-LOCO    | 48.3   | 54.4   | **+6.1** |
| Real-IAD      | 48.9   | 54.7   | **+5.8** |
| BTAD          | 49.2   | 53.8   | +4.6   |
| DGAD          | 46.5   | 51.2   | +4.7   |
| MPDD          | 48.1   | 52.9   | +4.8   |

---

## 🧠 Method Overview

CMD introduces four key innovations:

1. **Causal Latent Space Design**:  
   Structured encoder \( E_\phi: \mathcal{X} \rightarrow \mathcal{Z}_c \oplus \mathcal{Z}_n \) with HSIC + orthogonality losses.

2. **Meta-Pretraining via Normality Deviation**:  
   Simulates defects via procedural corruption (Perlin noise, masking, Gaussian blobs) on normal data.

3. **Few-Shot Adaptation with Causal Transfer**:  
   Meta-attention over base-class priors to infer causal importance vector \( w \).

4. **Interventional Similarity \( \mathcal{S}_{\text{do}} \)**:  
   Cosine similarity only on pruned causal subspace (\( \rho = 0.5 \)).

![CMD Workflow](assets/cmd_workflow.png)

---

## 📁 Code Structure

```
CMD/
├── configs/                  # YAML configs for datasets and training
├── data/                     # Dataset loading and corruption
│   ├── dataset_factory.py
│   ├── industrial_dataset.py
│   └── corruption.py
├── models/                   # Core CMD modules
│   ├── backbone.py
│   ├── causal_encoder.py
│   ├── decoder.py
│   ├── meta_transfer.py
│   └── detection_head.py
├── losses/                   # Custom losses
│   ├── vae_loss.py
│   ├── hsic.py
│   ├── ortho_loss.py
│   └── normality_deviation.py
├── utils/                    # Utilities
│   ├── metrics.py
│   ├── pca_normality.py
│   └── confounder_benchmark.py
├── scripts/                  # Training and evaluation
│   ├── train_base.py
│   ├── train_novel.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/yourname/CMD.git
cd CMD

# Create conda environment
conda create -n cmd python=3.9
conda activate cmd

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install perlin-noise  # For procedural corruption
```

---

## 🔁 Reproducibility Protocol

To ensure full reproducibility of all results reported in our manuscript, we provide complete details below. All experiments were conducted using the code in this repository.

### Random Seeds & Sampling
- **Support set sampling**: Fixed seed = 42  
- **Corruption generation**: Fixed seed = 45  
- **Training runs**: Five independent runs with seeds `{123, 456, 789, 999, 101}`  
- Results report **mean ± standard deviation** over these five runs.

### Hyperparameters
All values were selected via Bayesian optimization on MVTec-AD and used unchanged across datasets:
- HSIC weight (λ): **1.2**
- Sparsity ratio (ρ): **0.5**
- Normality threshold (κ): **3.0**
- Base learning rate: **0.006**
- Novel fine-tuning lr: **0.001**
- Batch size: **10**

See `configs/base.yaml` and `configs/novel.yaml` for full specifications.

### Baseline Evaluation
All baselines (TFA, Meta R-CNN, DeFRCN, FSCE, DLFG) were **re-implemented using official codebases** and evaluated under **identical conditions**: same backbone (ResNet-101), optimizer (SGD), resolution (512×512, 1024×1024 for Real-IAD), data splits, and random seeds.

### Hardware
- **GPUs**: 2 × NVIDIA A100 (40 GB memory each)
- **Software**: Python 3.8, PyTorch 1.12, CUDA 11.8

For full details, see the complete protocol at [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## 📥 Datasets

We evaluate on **7 industrial anomaly detection benchmarks**:

| Dataset       | Classes | Images  | Labels                     | Access |
|---------------|---------|---------|----------------------------|--------|
| **VisA**      | 12      | 10,821  | Bounding boxes             | [GitHub](https://github.com/amazon-science/spot-diff) |
| **MVTec-AD**  | 15      | 5,354   | Pixel masks                | [MVTec](https://www.mvtec.com/company/research/mvtec-ad/) |
| **MVTec-LOCO**| 5       | 3,644   | Pixel masks (logical/structural) | [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-loco/) |
| **Real-IAD**  | 8       | 151,050 | Bounding boxes             | [Email Request](https://realiad4ad.github.io/Real-IAD) |
| **BTAD**      | 3       | 2,540   | Pixel masks                | [Dataset Ninja](https://datasetninja.com/btad) |
| **DGAD**      | 10      | 17,100  | Pixel masks                | [GitHub](https://github.com/mala-lab/GGAD) |
| **MPDD**      | 6       | 1,346   | Bounding boxes             | [GitHub](https://github.com/stepanje/MPDD) |

> **Note**: Place all datasets under `/home/phd/datasets/` as:
> ```
> /home/phd/datasets/
> ├── VisA/
> ├── MVTec-AD/
> ├── MVTec-LOCO/
> ├── Real-IAD/
> ├── BTAD/
> ├── DGAD/
> └── MPDD/
> ```

---

## ▶️ Usage

### 1. Meta-Pretraining on Normal Data
```bash
python scripts/train_base.py --dataset mvtec_ad --epochs 20000
```

### 2. Few-Shot Fine-Tuning
```bash
python scripts/train_novel.py \
    --dataset mvtec_ad \
    --novel_class bottle \
    --k_shot 3 \
    --base_ckpt checkpoints/mvtec_ad_base_model.pth \
    --manifold_path checkpoints/mvtec_ad_manifold.pth
```

### 3. Evaluation
```bash
python scripts/evaluate.py \
    --dataset mvtec_ad \
    --k_shot 3 \
    --checkpoint checkpoints/mvtec_ad_bottle_3shot.pth \
    --base_ckpt checkpoints/mvtec_ad_base_model.pth \
    --manifold_path checkpoints/mvtec_ad_manifold.pth
```

---

## 📚 Citation

If you find CMD useful in your research, please cite our paper:

```bibtex
@article{cmd2025,
  title={Causal Meta-Disentanglement for Robust Few-Shot Object Detection},
  author={Saeed Iqbal},
  journal={IEEE Transactions on Industrial Informatics},
  year={2025}
}
```

Also cite the original datasets:

- **VisA**: [Liu et al., CVPR 2022](https://arxiv.org/abs/2206.08988)
- **MVTec-AD/LOCO**: [Bergmann et al., CVPR 2019, 2022](https://www.mvtec.com/company/research/datasets/)
- **Real-IAD**: [Wang et al., CVPR 2024](https://realiad4ad.github.io/Real-IAD)
- **BTAD**: [Mishra et al., ISIE 2021](https://datasetninja.com/btad)
- **DGAD**: [Wang et al., 2022](https://github.com/mala-lab/GGAD)
- **MPDD**: [Ježek et al., 2021](https://github.com/stepanje/MPDD)

---

## 📜 License

This project is released under the [Apache 2.0 License](LICENSE).

---

## 🙏 Acknowledgements

- This code builds upon [DLFG](https://ieeexplore.ieee.org/document/10558983) and [Meta R-CNN](https://openaccess.thecvf.com/content_ICCV_2019/html/Yan_Meta_R-CNN_Towards_General_Solver_for_Instance-Level_Low-Shot_Learning_ICCV_2019_paper.html).
- Industrial datasets are provided by their respective authors—thank you for enabling reproducible research!

---
