<p align="center">
  <h1 align="center">Spatio-Temporal Graph Neural Network for<br>12-Lead ECG Classification on PTB-XL</h1>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Dataset-PTB--XL-blue" alt="Dataset">
  <img src="https://img.shields.io/badge/Task-Multi--Label%20Classification-green" alt="Task">
</p>

<p align="center">
  <em>
  A Spatio-Temporal GNN that models the 12 standard ECG leads as nodes in an anatomically-informed graph,
  achieving <b>0.91 macro AUC-ROC</b> on PTB-XL diagnostic superclass classification with <b>13× fewer parameters</b>
  than a conventional 1D CNN baseline.
  </em>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
  - [Lead Graph Construction](#lead-graph-construction)
  - [Model Components](#model-components)
- [Dataset](#dataset)
- [Training Details](#training-details)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [References](#references)

---

## Overview

Conventional deep learning approaches for ECG classification (1D CNNs, RNNs) treat the 12 leads as flat input channels, discarding the known anatomical relationships between leads. This project explores a **Spatio-Temporal Graph Neural Network (ST-GNN)** that explicitly encodes the spatial structure of the standard 12-lead ECG system.

**Core idea:** Each ECG lead becomes a *node* in a graph. Edges connect anatomically related leads (e.g., limb leads I–II–III, the precordial chain V1–V6, and cross-region links). The model then alternates:

1. **Temporal Gated Convolutions** — learn waveform morphology (P-QRS-T complexes) along the time axis of each lead independently.
2. **Graph Convolutions** — propagate information across related leads via the fixed anatomical adjacency matrix.

This decoupled spatio-temporal design achieves strong multi-label classification performance while being dramatically more parameter-efficient than a 1D CNN baseline.

---

## Key Results

**Test set evaluation** on PTB-XL fold 10 (2,163 records):

### Per-Class AUC-ROC

| Class | Full Name               | AUC-ROC | Avg. Precision |  F1   |
|:------|:------------------------|:-------:|:--------------:|:-----:|
| NORM  | Normal ECG              | 0.936   |     0.907      | 0.849 |
| MI    | Myocardial Infarction   | 0.924   |     0.820      | 0.762 |
| STTC  | ST/T Change             | 0.930   |     0.818      | 0.757 |
| CD    | Conduction Disturbance  | 0.923   |     0.843      | 0.754 |
| HYP   | Hypertrophy             | 0.840   |     0.523      | 0.486 |
| **Macro** |                     | **0.911** |   **0.782**  | **0.722** |

### ST-GNN vs. 1D CNN Baseline

|                      | ST-GNN (ours) | 1D CNN Baseline |
|:---------------------|:-------------:|:---------------:|
| **Parameters**       |    427 K      |     5.79 M      |
| **Macro AUC-ROC**    |    0.911      |       —*        |
| **Architecture**     | Graph + Gated Conv | Deep ResNet-1D |
| **Spatial Modeling**  | Explicit graph adjacency | Implicit (flat channels) |

> *\*The 1D CNN baseline notebook is included for reference but was not trained to completion. The architecture (8-block ResNet-1D with 5.79M parameters) is provided for comparison.*

---

## Architecture

### Lead Graph Construction

The 12 ECG leads are arranged as nodes in a fixed, undirected graph with self-loops. Edges encode three types of anatomical relationships:

| Edge Group | Connections |
|:-----------|:------------|
| **Limb leads** | I↔II, I↔aVL, II↔III, II↔aVF, III↔aVF, aVR↔aVL, aVL↔aVF |
| **Precordial chain** | V1↔V2↔V3↔V4↔V5↔V6 |
| **Cross-region** | I↔V5, I↔V6, aVL↔V5, II↔V1, aVF↔V2 |

The adjacency matrix is symmetrically normalized using $\tilde{A} = D^{-1/2} A \, D^{-1/2}$ before being passed to graph convolution layers.

### Model Components

```
Input: (B, 12, 1000)           ← 12 leads × 10 seconds @ 100 Hz
        │
        ▼  unsqueeze
   (B, 1, 12, 1000)            ← (batch, channels, nodes, time)
        │
   ┌────┴────┐
   │ STGNNBlock 1 │  1 → 32 ch,  kernel=15
   └────┬────┘
   ┌────┴────┐
   │ STGNNBlock 2 │  32 → 64 ch, kernel=11
   └────┬────┘
   ┌────┴────┐
   │ STGNNBlock 3 │  64 → 96 ch, kernel=7
   └────┬────┘
        │
        ▼  AdaptiveAvgPool2d(1,1)
   (B, 96)
        │
        ▼  FC(96→64) → ReLU → Dropout → FC(64→5)
   (B, 5) logits
```

Each **STGNNBlock** follows a sandwich design:

```
TemporalGatedConv → GraphConv → BatchNorm → Dropout
       → TemporalGatedConv → BatchNorm → Dropout → (+Residual) → ReLU
```

| Component | Description |
|:----------|:------------|
| **TemporalGatedConv** | Conv2d with kernel $(1, K)$ over time, split into value/gate channels for GLU activation: $\text{out} = \text{value} \odot \sigma(\text{gate})$ |
| **GraphConv** | Separate linear projections for self-features and neighborhood-aggregated features: $W_s X + W_n (\tilde{A} X) + b$ |
| **Residual** | 1×1 Conv2d for channel matching when $C_{in} \neq C_{out}$ |

---

## Dataset

This project uses the [**PTB-XL**](https://physionet.org/content/ptb-xl/1.0.3/) dataset, a large publicly available electrocardiography dataset:

- **21,430 records** (10-second, 12-lead ECGs) after filtering for diagnostic labels
- **5 diagnostic superclasses**: NORM, MI, STTC, CD, HYP
- **Multi-label**: a single record can belong to multiple classes
- **Official stratified folds** (`strat_fold`):
  - Train: folds 1–8 (17,111 records)
  - Validation: fold 9 (2,156 records)
  - Test: fold 10 (2,163 records)

| Superclass | Records | Description |
|:-----------|:-------:|:------------|
| NORM | 9,528 | Normal ECG |
| MI | 5,486 | Myocardial Infarction |
| STTC | 5,250 | ST/T Change |
| CD | 4,907 | Conduction Disturbance |
| HYP | 2,655 | Hypertrophy |

---

## Training Details

| Hyperparameter | Value |
|:---------------|:------|
| Sampling rate | 100 Hz |
| Sequence length | 1,000 samples (10 s) |
| Batch size | 32 |
| Epochs | 15 |
| Optimizer | AdamW (lr=5×10⁻⁴, weight decay=1×10⁻⁴) |
| LR scheduler | OneCycleLR (cosine, 10% warmup) |
| Loss | BCEWithLogitsLoss with power-scaled pos_weight ($\alpha = 1.2$) |
| Dropout | 0.25 |
| Mixed precision | AMP (autocast + GradScaler) |
| Gradient clipping | max_norm = 1.0 |

**Data augmentation** (training only):
- Gaussian noise ($p = 0.5$, $\sigma = 0.02$)
- Random amplitude scaling ($p = 0.5$, scale ∈ [0.85, 1.15])
- Random temporal shift ($p = 0.3$, ±100 samples circular roll)

**Per-class decision thresholds** are tuned on the validation set by sweeping [0.10, 0.90] to maximize F1 per class, rather than using a fixed 0.5 threshold.

---

## Repository Structure

```
ST-GNN-for-PTB-XL-Dataset/
├── README.md                  ← You are here
├── .gitignore
├── st-gnn-ptb-xl.ipynb        ← Main ST-GNN notebook (training + evaluation)
└── ptbxl_1dcnn.ipynb           ← 1D CNN baseline (architecture reference)
```

| File | Description |
|:-----|:------------|
| `st-gnn-ptb-xl.ipynb` | End-to-end ST-GNN pipeline: data loading, graph construction, model definition, training, evaluation with full test results, and model bundling for inference |
| `ptbxl_1dcnn.ipynb` | 1D ResNet CNN baseline for comparison — demonstrates the conventional approach of treating leads as flat channels |

---

## Getting Started

### Prerequisites

```bash
pip install torch numpy pandas matplotlib seaborn tqdm wfdb scikit-learn
```

### Dataset

1. Download PTB-XL from [Kaggle](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset) or [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
2. Extract into the `archive/` directory (gitignored)
3. Update `DATA_DIR` in the notebook to point to the extracted dataset path

### Running

Open `st-gnn-ptb-xl.ipynb` in Jupyter or run on [Kaggle](https://www.kaggle.com/) with GPU acceleration. The notebook is self-contained — simply run all cells sequentially.

---

## References

1. Wagner, P., et al. "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data* 7, 154 (2020). [doi:10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)

2. Strodthoff, N., et al. "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL." *IEEE Journal of Biomedical and Health Informatics* 25.5 (2021). [doi:10.1109/JBHI.2020.3022989](https://doi.org/10.1109/JBHI.2020.3022989)

3. Ribeiro, A.H., et al. "Automatic diagnosis of the 12-lead ECG using a deep neural network." *Nature Communications* 11, 1760 (2020). [doi:10.1038/s41467-020-15432-4](https://doi.org/10.1038/s41467-020-15432-4)

4. Yu, B., Yin, H., & Zhu, Z. "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Flow Forecasting." *IJCAI* (2018). [doi:10.24963/ijcai.2018/505](https://doi.org/10.24963/ijcai.2018/505)

5. Yan, G., et al. "Spatio-temporal graph neural network for ECG arrhythmia classification." *PMC* (2025). [PubMed:41176406](https://pubmed.ncbi.nlm.nih.gov/41176406/)

---

<p align="center">
  <sub>Built as a deep learning course project · Cairo University</sub>
</p>