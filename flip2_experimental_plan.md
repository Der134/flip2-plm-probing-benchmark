# FLIP2 Benchmark Experimental Architecture & Protocol

## 1. Executive Summary & Research Objectives

The **FLIP2 (Fitness Landscapes for Protein Sequences)** benchmark provides standardized datasets for evaluating protein language model (PLM) representations on fitness prediction tasks. 

This experimental protocol establishes a systematic framework for evaluating pre-trained ESM-2 model scale variants ($8\text{M}$, $35\text{M}$, $150\text{M}$, $650\text{M}$, $3\text{B}$) and ESMC-600M under frozen representation regimes.

### Core Objectives
1. **Representation Benchmark**: Quantify the downstream predictive capacity ($R^2$, Spearman $\rho$, MSE) of frozen PLM embeddings against a standardized One-Hot Baseline across diverse fitness landscapes (`amylase`, `rhomax`).
2. **Data Efficiency Scaling**: Evaluate model performance under data-constrained regimes by subsampling training sets at fractions $f \in \{0.10, 0.20, 0.40, 0.60, 0.80, 1.00\}$.
3. **Uncertainty Quantification (UQ)**: Estimate epistemic uncertainty via a 5-fold Deep Ensemble MLP head, measuring calibration metrics (Negative Log-Likelihood, Expected Calibration Error) and rejection-curve RMSE degradation.
4. **Compute Efficiency & Dual-Account Optimization**: Execute all GPU feature extraction within a target budget of $\le 10\text{ Compute Units (CU)}$ on Google Colab through dual-account workspace synchronization and dynamic token-budget batching.

---

## 2. Directory & Dual-Account Synchronization Protocol

To ensure seamless execution across multiple Google Colab accounts without duplicate file generation or storage quota waste, both accounts access a single shared Google Drive directory.

```
/content/drive/MyDrive/flip2_benchmark/
├── .workspace_id                     # Sentinel verification file
├── data/
│   ├── raw/                          # Original FLIP2 dataset files (.csv)
│   └── processed/
│       ├── pilot_manifest.json       # Dataset metadata & SHA-256 fingerprints
│       └── unique_sequences.csv      # Deduplicated global sequence registry
├── cache/
│   └── embeddings/                   # Atomic .npz feature stores
│       ├── esm2_8m__amylase__<sha>.npz
│       ├── esm2_3b__rhomax__<sha>.npz
│       └── ...
└── outputs/
    ├── figures/                      # Publication-grade evaluation plots
    └── tables/                       # Summary CSV performance metrics
```

### Sentinel Hard-Validation Protocol
Before executing any GPU workload, notebooks enforce hard validation of `.workspace_id` containing `flip2-shared-prod-2026`:
* If missing, execution terminates immediately to prevent silent fallback to a local isolated directory.
* Resolves the FUSE shortcut path (`/content/drive/.shortcut-targets-by-id/...`) to guarantee identical underlying storage for both Account 1 (Owner) and Account 2 (Shared User).

---

## 3. Pipeline Architecture

```
+-----------------------------------------------------------------------+
| Stage 1: Data Prep & Baselines (CPU)                                  |
| - Validate sequence cleanliness & normalize target fitness values    |
| - Perform global sequence deduplication & SHA-256 hashing             |
| - Construct One-Hot baseline representations                          |
| - Export pilot_manifest.json & unique_sequences.csv                   |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| Stage 2: Frozen Embedding Extraction (GPU Dispatch)                   |
| - L4 GPU Instance (~4.7 CU/h): ESM-2 (8M, 35M, 150M, 650M)           |
| - G4 GPU Instance (~10.5 CU/h): ESM-2 (3B)                            |
| - Dynamic Token-Budget Batching + Native PyTorch SDPA (bf16)          |
| - Exclude BOS/EOS/PAD tokens during residue mean-pooling             |
| - Write atomic .npz files + invoke drive.flush_and_unmount()         |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| Stage 3: Probing, Data Efficiency & UQ Analysis (CPU)                 |
| - Align deduplicated embeddings with split indices                   |
| - Train Ridge Regression ($\alpha \in [10^{-3}, 10^{3}]$ grid search)  |
| - Evaluate Data Efficiency curves (10% to 100% training subsets)      |
| - Train 5x Deep Ensemble MLP heads for epistemic variance $\sigma^2$  |
| - Generate publication-ready figures & metric CSV reports             |
+-----------------------------------------------------------------------+
```

---

## 4. Hardware Allocation & Compute Budget Guardrails

| Stage | Resource Tier | Execution Target | Estimated Time | Estimated Cost |
| :--- | :--- | :--- | :--- | :--- |
| **01 Data Prep** | CPU Runtime | Sequence cleaning & One-Hot Baseline | $\sim 3\text{ min}$ | $0.0\text{ CU}$ |
| **02 Small PLMs** | L4 GPU ($24\text{ GB}$) | ESM-2 ($8\text{M}$, $35\text{M}$, $150\text{M}$, $650\text{M}$) | $\sim 45\text{ min}$ | $\sim 3.5\text{ CU}$ |
| **02 Large PLM** | G4 GPU ($96\text{ GB}$) | ESM-2 ($3\text{B}$) | $\sim 20\text{ min}$ | $\sim 3.5\text{ CU}$ |
| **03 Probing & UQ**| CPU Runtime | Ridge CV, Data Efficiency, 5x Ensemble | $\sim 10\text{ min}$ | $0.0\text{ CU}$ |
| **Total Pilot** | | | $\sim 78\text{ min}$ | **$\sim 7.0\text{ CU}$** |

### Anti-Waste Guardrails
1. **G4 Small Model Intercept**: Attempts to run models $\le 650\text{M}$ on a G4 instance ($10.5\text{ CU/h}$) trigger an explicit error unless `ALLOW_SMALL_ON_G4 = True` is set.
2. **First-Batch Peak Memory Trigger**: Sequences are sorted in descending order of length ($L_{\text{max}}$ first). If a memory error occurs, it triggers immediately on the first batch rather than failing mid-run.
3. **Atomic File Serialization**: Features are written to a temporary `.tmp` file and renamed atomically to prevent corrupted cache files upon timeout or disconnect.