> **Explainable Vision Transformers for *Candida* Species Identification from Brightfield Microscopy**  
> *IEEE World Congress on Computational Intelligence (WCCI) 2026*

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Reproducibility: Step-by-Step](#reproducibility-step-by-step)
  - [Step 1 — Canonical Data Split](#step-1--canonical-data-split)
  - [Step 2 — Preprocessing](#step-2--preprocessing)
  - [Step 3 — Model Training (Google Colab)](#step-3--model-training-google-colab)
  - [Step 4 — CV Statistics](#step-4--cv-statistics)
  - [Step 5 — XAI Analysis](#step-5--xai-analysis)
- [Configuration](#configuration)
- [Key Design Decisions](#key-design-decisions)
- [Hardware and Compute](#hardware-and-compute)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## Overview

Invasive fungal infections (IFIs) cause over **1.5 million deaths annually**, with *Candida* designated a WHO priority pathogen. A critical diagnostic bottleneck is the rapid, reliable differentiation of *Candida albicans* from *Candida glabrata* directly from brightfield microscopy. These species are therapeutically decisive: *C. glabrata* frequently exhibits acquired resistance to azole antifungals, necessitating divergent treatment protocols. However, morphological ambiguity, overlapping blastospore size and budding patterns, causes substantial inter-observer variability in direct microscopic examination.

This repository provides the **fully reproducible pipeline** for:

1. **Dataset curation** — A novel dataset of 3,731 brightfield microscopy patches from 60 unique biological slides, curated with strict slide-level biological isolation
2. **SimpleViT classification** — A data-efficient Vision Transformer achieving 92.86% slide-level balanced accuracy on a held-out test set, validated under 30 × 5-fold CV.
3. **Explainability analysis (XAI)** — A forensic investigation using UMAP latent manifold analysis and Attention Rollout that identifies two distinct error regimes: morphological mimicry in False Positives and signal attenuation in False Negatives.

---

## Results

### Cross-Validation Performance (30 × 5-fold Repeated CV)

| Metric | Patch-Level | Slide-Level |
|--------|-------------|-------------|
| Balanced Accuracy | 0.8287 ± 0.0330 [0.816, 0.841] | 0.9117 ± 0.0374 [0.898, 0.926] |
| MCC | 0.6562 ± 0.0666 [0.631, 0.681] | 0.8319 ± 0.0715 [0.805, 0.859] |
| F1-Score (W) | 0.8249 ± 0.0350 [0.812, 0.838] | 0.8992 ± 0.0490 [0.881, 0.918] |
| AUC | 0.9067 ± 0.0359 [0.893, 0.920] | 0.9618 ± 0.0340 [0.949, 0.975] |

*Mean ± Std (between-iteration), 95% CI via t_{N−1} distribution (N = 30 iterations). Computed via two-level hierarchical aggregation: fold scores averaged within each iteration first, then aggregated across iterations to account for data overlap.*

### Held-Out Test Set (N = 12 slides, 821 patches)

95% CIs via cluster-robust non-parametric bootstrapping (B = 1,000): slides as clusters for patch-level metrics; singleton resampling for slide-level metrics.

| Metric | Patch-Level | Slide-Level |
|--------|-------------|-------------|
| Balanced Accuracy | 0.8439 [0.745, 0.897] | **0.9286** [0.786, 1.000] |
| MCC | 0.7025 [0.516, 0.816] | 0.8452 [0.529, 1.000] |
| F1-Score (W) | 0.8326 [0.729, 0.926] | 0.9172 [0.733, 1.000] |
| AUC | 0.9422 [0.847, 0.985] | 0.9714 [0.852, 1.000] |

**Patch-level confusion matrix (N = 821 patches):**

|  | Predicted *C. albicans* | Predicted *C. glabrata* |
|--|--|--|
| **Actual *C. albicans*** | 313 | 127 |
| **Actual *C. glabrata*** | 9 | 372 |

**Slide-level confusion matrix (N = 12 slides):**

|  | Predicted *C. albicans* | Predicted *C. glabrata* |
|--|--|--|
| **Actual *C. albicans*** | 6 | 1 |
| **Actual *C. glabrata*** | 0 | 5 |

Mean-pooling aggregation suppresses local patch noise: MCC improves by **+26.8%** (0.656 → 0.832) upon aggregation, with a Coefficient of Variation of ≈ 4.0%.

### XAI Findings — Attention Rollout Topology

| Error Group | n | Gini µ (focality) | Entropy µ (diffuseness) | Interpretation |
|-------------|---|-------------------|--------------------------|----------------|
| True Positive | 372 | 0.465 | — | Focal, semantically grounded attention on cellular targets |
| **False Positive** | **127** | **0.460** | — | **Morphological mimicry: confident focal attention on yeast-like textures** |
| False Negative | 9 | — | 5.94 | Signal attenuation: fragmented, entropic attention mass |

Welch's t-test (TP vs FP Gini): **p ≈ 0.027**, Cohen's d ≈ 0.22 (negligible effect size). The near-identical focality of TP and FP confirms the model treats false positives as confident positives — the primary failure mode is **systematic confident misidentification**, not stochastic noise.

The class-stratified permutation test (B = 1,000) on the UMAP manifold rejects the null hypothesis of random error distribution (p < 0.001), with observed slide-level error variance σ²_obs ≈ 0.069, approximately **3.5× higher than the null expectation** — errors are structurally bound to specific biological samples.

---

## Dataset

| Class | Slides | Train Patches | Test Patches | Total |
|-------|--------|---------------|--------------|-------|
| *C. albicans* | 33 | 1,616 | 440 | 2,056 |
| *C. glabrata* | 27 | 1,294 | 381 | 1,675 |
| **Total** | **60** | **2,910** | **821** | **3,731** |

### Acquisition Protocol

| Parameter | Value |
|-----------|-------|
| Microscope | Zeiss Axio Observer |
| Objective | Plan-Apochromat 63×/1.4 Oil DIC |
| Camera | Prime 95B monochrome (12-bit) |
| Pixel spacing | 0.175 µm |
| Illumination | Brightfield |
| Mounting | 1.7% agarose pad |
| Culture medium | Synthetic Complete (pH 6.5), 35°C, OD₆₀₀ = 3.0 |
| FOVs per slide | ≈ 62 (random, fixed focus) |
| Images/slide: *C. albicans* | 62.3 ± 35.2 |
| Images/slide: *C. glabrata* | 62.0 ± 15.7 |

The physical slide is the **primary independent biological replicate**. All patches from a single slide are assigned exclusively to either the training or test partition. The individual FOV is the dependent observational unit.

### Availability

**The dataset will be made publicly available via an open-access repository upon paper acceptance.** Please contact the corresponding author in the meantime.

### Data Format

After preprocessing, images are stored as `.npy` float32 arrays of shape `(384, 384)` with values in [0, 1]. The directory layout follows the `torchvision.datasets.DatasetFolder` convention, enabling direct use with `NPYImageFolder` (see `training/training_utils.py`):

```
preprocessed/
├── metadata.json                           # {method, bit_depth, image_size, baSiC_used}
├── Candida albicans/
│   ├── 20250101_alb_A_patch_0001.npy
│   └── ...
└── Candida glabrata/
    ├── 20250101_gla_B_patch_0001.npy
    └── ...
```

**Filename convention:** `YYYYMMDD_speciescode_letter_patch_NNNN.npy`

The slide identifier is the first three underscore-delimited components: `YYYYMMDD_speciescode_letter` (e.g. `20250923_alb_K`).

---

## Model Architecture

We adopt **SimpleViT** (Beyer et al., 2022), a streamlined Vision Transformer optimised for data-efficient classification. It refines the canonical ViT (Dosovitskiy et al., 2021) by replacing learnable positional embeddings and the CLS token with fixed 2D sinusoidal positional encodings and Global Average Pooling.

### Configuration

| Component | Parameter | Value |
|-----------|-----------|-------|
| Input | Resolution | 384 × 384 px (grayscale, C = 1) |
| Tokenisation | Patch size P | 16 → **576 tokens** |
| Encoder | Layers L | 8 |
| Encoder | Embedding dim D | 256 |
| Encoder | Attention heads | 6 |
| Encoder | FFN expansion ratio | 2 → MLP dim = 512 |
| Positional encoding | Type | Fixed 2D sinusoidal (E_sincos) |
| Classification head | Pooling | Global Average Pooling (no CLS token) |
| Classification head | Output | Linear(256 → 2) |

### Patch Projection

The dual-normalisation patch projection bounds signal variance at initialisation:

```
z₀ = LN(Linear(LN(x_patches))) + E_sincos
```

### Why SimpleViT for this task

CNNs are constrained by local receptive fields, they cannot naturally model long-range dependencies between dispersed spores or fragmented hyphal structures. ViTs process images as patch sequences, maintaining a global contextual scope from the first layer. The SimpleViT's Global Average Pooling (replacing the CLS token) forces semantic attention to be distributed across all 576 patch positions, which directly enables the high-fidelity Attention Rollout maps analysed in Section VI of the paper.

---

## Repository Structure

```
fungi-vit-xai/
│
├── README.md
├── LICENSE                             # MIT
├── .gitignore
├── requirements.txt
│
├── data/                               # ── PHASE 1: Dataset Curation ──────────────────────
│   ├── canonical_splits.py             # Slide-stratified split → split_indices.json
│   ├── preprocessing.py                # BaSiCPy illumination correction + normalisation
│   └── processing_helper.py            # CLI orchestrator: split + preprocess → .npy dataset
│
├── training/                           # ── PHASE 2: Model Training ─────────────────────────
│   ├── vit_fungi.ipynb                 # ★ PRIMARY ARTIFACT — open and run in Google Colab
│   ├── config.py                       # All hyperparameters — single source of truth
│   ├── training_logic.py               # ModelTrainer, training loop, determine_optimal_epochs
│   ├── training_utils.py               # NPYImageFolder, slide-level CV folds, bootstrap CI
│   ├── checkpoints.py                  # CheckpointManager: atomic saves, full resumability
│   └── plotting_utils.py               # Training curves, confusion matrix, ROC/AUC
│
├── analysis/                           # ── PHASES 3 & 4: Post-Training Analysis ────────────
│   ├── cv_statistics.py                # Hierarchical CV stats, LOCF curves, LaTeX table
│   └── xai/                            # Explainability sub-pipeline
│       ├── analysis_attention.py       # ★ Main XAI entry point (CLI)
│       ├── extractor.py                # Embeddings, Attention Rollout, Integrated Gradients
│       ├── embeddings.py               # UMAP/t-SNE, trustworthiness, permutation test
│       ├── stratified_sampling.py      # Gini, Shannon entropy, Welch tests, Cohen's d
│       ├── visualization.py            # Attention heatmaps, slide-conditioned error plots
│       ├── helper.py                   # Slide ID parsing, provenance mapping, validation
│       ├── config.py                   # Copy of training/config.py (self-contained XAI run)
│       └── training_utils.py           # Copy of training/training_utils.py (inference subset)
│
├── configs/                            # Generated experiment configs (JSON, at runtime)
└── outputs/                            # Generated model outputs (at runtime; not committed)
```

> **Note on shared files.** `analysis/xai/config.py` and `analysis/xai/training_utils.py`
> are intentional copies of their counterparts in `training/`, kept self-contained so the
> XAI pipeline can be run independently without modifying `PYTHONPATH`. If you modify one,
> update the other accordingly.

---

## Installation

### System Requirements

- Python ≥ 3.10
- **Training:** CUDA-capable GPU

### Environment Setup

```bash
git clone https://github.com/Rodrigo2003-PT/simpleViT_fungi.git
cd simpleViT_fungi

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt
```

### Dependency Notes

**BaSiCPy** (`basicpy`) requires `jax` as a computational backend. If installation fails, consult the [BaSiCPy installation guide](https://github.com/peng-lab/BaSiCPy). The `baseline` preprocessing method (linear normalisation only) works without BaSiCPy.

**vit-pytorch** provides the SimpleViT implementation. The XAI extractor (`extractor.py`) relies on the internal API of `lucidrains/vit-pytorch` — specifically the attribute names `to_qkv`, `attend`, `heads`, `scale`, and `norm` in each attention module. 

**UMAP** (`umap-learn`) is only required for the XAI embedding analysis phase and can be omitted if only running preprocessing and training.

---

## Reproducibility: Step-by-Step

The pipeline has **five sequential steps**. Steps 1, 2, 4, and 5 run locally as CLI scripts. Step 3 runs in Google Colab.

```
Raw 12-bit TIFF dataset (60 slides)
          │
          ▼
[Step 1]  data/canonical_splits.py
          │   Slide-stratified constrained random search (seed = 42)
          ▼
          split_indices.json  ← commit this; it is the reproducibility anchor
          │
          ▼
[Step 2]  data/processing_helper.py
          │   BaSiCPy correction (train-only) + Lanczos resize → float32 .npy
          ▼
          preprocessed_384.zip  +  metadata.json
          │
          │   ← upload to Google Drive →
          ▼
[Step 3]  training/vit_fungi.ipynb          (Google Colab, T4 GPU)
          │   Phase 1: 30 × 5-fold CV
          │   Phase 2: optimal epoch determination
          │   Phase 3: final training on full cohort
          │   Phase 4: held-out test evaluation + bootstrap CI
          ▼
          final_model_standard.pth
          cv_results_standard.csv
          fold_histories/
          test_results_standard.json
          │
          │   ← download outputs from Google Drive →
          ▼
[Step 4]  analysis/cv_statistics.py
          │   Hierarchical aggregation + LOCF curves + LaTeX table
          ▼
          results_table.tex  +  learning_curves.png
          │
          ▼
[Step 5]  analysis/xai/analysis_attention.py
              UMAP + Attention Rollout + (optional) Integrated Gradients
          ▼
          umap_comprehensive_test.png
          rollout_stats_test.json
          rep_{TP,FP,FN}_rollout_test.png
```

---

### Step 1 — Canonical Data Split

Generates `split_indices.json` to guarantee slide-level biological isolation.

```bash
python data/canonical_splits.py \
    --tiff-dir  /path/to/raw_tiff_dataset \
    --output    split_indices.json \
    --test-size 0.15 \
    --seed      42
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--tiff-dir` | Root of raw TIFF acquisitions (class subdirs). Accepts a `.zip` archive. |
| `--output` | Output path for the JSON split file (default: `split_indices.json`) |
| `--test-size` | Fraction of slides assigned to the test partition (default: `0.15`) |
| `--seed` | Random seed for the constrained search (default: `42`) |

**Algorithm.** The script enumerates all TIFF images across class subdirectories, groups them by slide identifier (the first three underscore-delimited filename components: `YYYYMMDD_speciescode_letter`), and runs a constrained randomised search over up to 2,000 candidate splits. A valid split must simultaneously satisfy:

- **Slide-level stratification** — each class contributes approximately `test_size` of its slides to the test partition
- **Image-level global class ratio match** within ±2% tolerance — KL-divergence minimisation (δ < 0.02)

**Key fields in `split_indices.json`:**

```json
{
    "train_base_paths":   ["Candida albicans/20250101_alb_A_patch_0001", ...],
    "test_base_paths":    ["Candida albicans/20250101_alb_B_patch_0001", ...],
    "classes":            ["Candida albicans", "Candida glabrata"],
    "random_seed":        42,
    "test_size":          0.15,
    "total_samples":      3731,
    "train_count":        2910,
    "test_count":         821,
    "class_distribution": {"train": {...}, "test": {...}},
    "drift_metrics": {
        "global_ratio":   0.551,
        "test_ratio":     0.536,
        "drift_achieved": 0.015
    }
}
```

> **Important.** The `split_indices.json` committed to this repository is the **canonical split used in all paper results**. Do not regenerate it if you want to reproduce the paper's numbers exactly — use the committed file directly.

---

### Step 2 — Preprocessing

Converts raw 12-bit TIFF images to normalised float32 `.npy` arrays at 384 × 384 pixels. Two methods are available:

| Method | Description | Recommended for |
|--------|-------------|-----------------|
| `baseline` | Global linear normalisation only: I_norm = I_raw / 4095 → [0, 1] | ✓ Paper results |
| `optimized` | Baseline + BaSiCPy flatfield/darkfield illumination correction | Ablation |

```bash
python data/processing_helper.py \
    --input         /path/to/raw_tiff_dataset \
    --output        ./preprocessed \
    --method        optimized \
    --image-size    384 \
    --bit-depth     12 \
    --split-json    split_indices.json \
    --seed          42 \
    --basic-samples 100 \
    --create-zip
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | — | Raw TIFF dataset directory or `.zip` archive |
| `--output` | — | Output directory |
| `--method` | `baseline` | `baseline` or `optimized` |
| `--image-size` | `384` | Target square resolution (px) |
| `--bit-depth` | `12` | Sensor bit depth of raw images |
| `--split-json` | — | **Required.** Path to `split_indices.json` |
| `--basic-samples` | `100` | Images used to fit BaSiCPy (100–200 recommended) |
| `--seed` | `42` | Random seed for BaSiCPy sample selection |
| `--create-zip` | flag | Zip the output directory on completion |
| `--force` | flag | Reprocess even if `.npy` files already exist |

**Why `--split-json` is mandatory.** BaSiCPy is fitted **exclusively on training images**. This prevents any signal from test images leaking into the illumination correction model.

**Quality metrics reported at runtime:**

- Background std before/after correction (empty-field mask at 90th percentile threshold)
- Correction score Γ'(I_corr): ratio of mean pairwise differences after/before correction
  - Γ' < 1 → correction improves consistency ✓
  - Γ' > 1.05 → preprocessing **aborts** with a `RuntimeError` to protect data integrity
- Per-species, per-slide breakdown of the BaSiCPy fitting set

**Output structure:**

```
preprocessed/
├── metadata.json            # {method, bit_depth, image_size, n_channels, baSiC_used}
├── Candida albicans/
│   └── *.npy
└── Candida glabrata/
    └── *.npy
```

After running with `--create-zip`, this directory is compressed into `preprocessed_384.zip` for upload to Google Drive before Step 3.

---

### Step 3 — Model Training (Google Colab)

The training pipeline was developed and executed on **Google Colab (T4 GPU)** and is distributed as the original Colab notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/fungi-vit-xai/blob/main/training/vit_fungi.ipynb)

#### Google Drive Setup

Arrange your Google Drive **before** opening the notebook:

```
MyDrive/colab/vit_fungi/
│
├── preprocessed/
│   ├── preprocessed_384.zip        ← output of Step 2 (--create-zip)
│   └── split_indices.json          ← output of Step 1
│
├── scripts/                        ← upload all .py files from training/
│   ├── config.py
│   ├── training_logic.py
│   ├── training_utils.py
│   ├── checkpoints.py
│   └── plotting_utils.py
│
├── output/standard/                ← created automatically
└── checkpoints/standard/           ← created automatically
```

Then open `vit_fungi.ipynb` in Colab and run all cells. The notebook mounts Drive, copies scripts to `/content/scripts/`, extracts `preprocessed_384.zip` to `/content/data/`, installs `vit-pytorch`, and enters the four-phase pipeline.

#### Four-Phase Resumable Pipeline

The notebook implements a **state machine with full checkpoint resumability**. If the Colab session disconnects at any point, re-running the notebook automatically resumes from the last saved checkpoint with no loss of training progress.

**Phase 1 — Cross-Validation (30 iterations × 5 folds = 150 total folds)**

- Slide-level stratified k-fold: `StratifiedKFold` applied at the slide level, then mapped to image indices. Quality constraints enforced per fold: ≥ 4 slides per class, slide balance ratio ≥ 0.5; up to 50 retry attempts with perturbed seeds
- Per-fold normalisation: mean and std computed strictly from training-fold image indices — no leakage from validation
- Model initialisation seed derived deterministically from iteration seed via `SeedSequence`
- **Model selection (lexicographic)** via `LexBestModelTracker`: (1) maximise slide-level Balanced Accuracy; (2) minimise validation loss as tiebreaker at ε = 10⁻⁶
- **Early stopping**: patience = 20 on the EMA-smoothed (α = 0.3) slide-level BA signal, via `ExponentialMovingAverage` + `PatienceTracker`
- **Checkpointing**: atomic saves every 25 epochs, on early stopping, and at fold completion — full RNG state (PyTorch + CUDA + NumPy + Python `random`) preserved

**Phase 2 — Analysis**

Aggregates 150 fold summaries to determine optimal final training duration. Uses the median of per-iteration mean best-epoch values — robust to outlier folds from early stopping variance.

**Phase 3 — Final Training**

Retrains on the full training cohort (48 slides, 2,910 patches) for the determined number of epochs. Normalisation statistics computed from the full training set. Final model saved as `final_model_standard.pth` with config, normalisation stats, and `class_to_idx` embedded in the checkpoint.

**Phase 4 — Evaluation**

Evaluates on the held-out test set (12 slides, 821 patches):

- Patch-level inference with `torch.inference_mode()`, softmax probabilities
- Slide-level aggregation: mean probability pooling across all patches from a slide, then argmax
- 95% CI via cluster-robust bootstrap (B = 1,000)
- Per-class precision, recall, F1 at both levels
- Full results saved to `test_results_standard.json`

**Key outputs** (in `MyDrive/colab/vit_fungi/output/standard/`):

| File | Description |
|------|-------------|
| `final_model_standard.pth` | Trained model with config and normalisation stats embedded |
| `cv_results_standard.csv` | One row per fold: all metrics at the best epoch |
| `checkpoints/fold_histories/` | Per-fold training histories (JSON) for LOCF curve generation |
| `test_results_standard.json` | Full test evaluation with CIs and per-class metrics |
| `config_standard.json` | Exact hyperparameter config used |

### Step 4 — CV Statistics

Computes Table II and Figure 2 of the paper from cross-validation outputs.

```bash
python analysis/cv_statistics.py \
    --csv-path      /path/to/cv_results_standard.csv \
    --history-dir   /path/to/fold_histories \
    --output-dir    ./outputs/stats \
    --n-iterations  30 \
    --n-folds       5
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv-path` | — | **Required.** Path to `cv_results_standard.csv` |
| `--history-dir` | — | **Required.** Path to the `fold_histories/` directory |
| `--output-dir` | `./stats_output` | Directory for all output files |
| `--n-iterations` | `30` | Number of CV iterations |
| `--n-folds` | `5` | Number of folds per iteration |
| `--confidence-level` | `0.95` | CI confidence level |

1. **Within each of the 30 iterations**: average the k = 5 fold scores → one iteration mean. Collapses within-iteration fold correlation
2. **Across the 30 iteration means**: global mean, between-iteration std (ddof = 1), 95% CI via t_{N−1} (N = 30 df)

This is the correct treatment for repeated k-fold CV. Treating all 150 fold scores as independent observations would overestimate precision by a factor of √5 ≈ 2.24.

**LOCF learning curves:**

Fold training histories have heterogeneous lengths due to early stopping. Last Observation Carried Forward (LOCF) imputation extends each fold's curve to T_max by repeating the last observed value. Fold curves within each iteration are then averaged (k = 5), followed by between-iteration mean ± 1σ aggregation (N = 30). This produces Figure 2 in the paper.

**Outputs:**

| File | Paper figure / table |
|------|----------------------|
| `cv_results_cleaned.csv` | — |
| `results_table.tex` | Table II |
| `learning_curves.png` | Figure 2 |
| `metric_distribution_slide_level.png` | Supplementary |
| `metric_distribution_patch_level.png` | Supplementary |

---

### Step 5 — XAI Analysis

The XAI pipeline runs locally on the trained model. GPU recommended for Integrated Gradients; CPU is sufficient for UMAP and Rollout.

#### Full command (UMAP + Attention Rollout)

```bash
python analysis/xai/analysis_attention.py \
    --preprocessed-root   /path/to/preprocessed \
    --dataset-subdir      . \
    --output-root         /path/to/outputs \
    --split-file          split_indices.json \
    --experiment          standard \
    --split               test \
    --extract-attention \
    --run-embedding-analysis \
    --compute-attention-rollout \
    --rollout-discard-ratio 0.9
```

**Note on model path resolution.** The pipeline resolves the final model from:
```
{output-root}/{experiment}/final_model_{experiment}.pth
```
Place the downloaded `final_model_standard.pth` at `./outputs/standard/final_model_standard.pth`, or set `--output-root` to the directory where you stored it.

**All arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--preprocessed-root` | `./preprocessed` | Root of the preprocessed `.npy` dataset |
| `--dataset-subdir` | `fungi` | Subdirectory inside `preprocessed-root` |
| `--output-root` | `./outputs` | Root for all analysis outputs |
| `--split-file` | `./split_indices.json` | Path to canonical split file |
| `--experiment` | `standard` | Experiment name (must match training output) |
| `--split` | `test` | Split to analyse: `train` or `test` |
| `--batch-size` | `8` | Inference batch size |
| `--force-recompute` | flag | Re-extract embeddings even if `.npz` already exists |
| `--extract-attention` | flag | Extract and save full spatial attention tensors per layer |
| `--attention-layers` | `None` (all) | Specific layer indices to extract (e.g. `6 7`) |
| `--run-embedding-analysis` | flag | UMAP projection + slide-aware metrics |
| `--compute-attention-rollout` | flag | Attention Rollout + Gini/entropy metrics |
| `--rollout-discard-ratio` | `0.9` | Bottom fraction of attention links to prune per layer |
| `--compute-integrated-gradients` | flag | Integrated Gradients attribution maps (slow) |
| `--ig-steps` | `50` | Number of integration steps along the path |
| `--attribution-aggregate-method` | `l2norm` | IG aggregation over embedding dim: `l2norm`, `meanabs`, `sumabs` |
| `--attribution-target-class` | `None` | Target class for IG; `None` uses the predicted class |

#### UMAP Latent Manifold Analysis

Activated by `--run-embedding-analysis`. Extracts `[N, 256]` pooled embeddings, projects to ℝ² via UMAP (n_neighbors = 15, min_dist = 0.1, seed = 42), and reports:

- **Trustworthiness T(k = 5)**: fraction of k-nearest high-dimensional neighbours preserved in the projection (paper: T ≈ 0.97)
- **Silhouette score** on ground truth and predicted labels
- **Class-stratified permutation test** (B = 1,000): slide-level error rate variance compared against a null distribution constructed by permuting errors *within class*, controlling for species-level difficulty. Reports p-value and ratio of observed to null variance (paper: p < 0.001, ratio ≈ 3.5×)

#### Attention Rollout

Activated by `--extract-attention --compute-attention-rollout`. Implements Abnar & Zuidema (2020) adapted for Global Average Pooling:

1. Forward hooks re-compute `softmax(QK^T / scale)` from `to_qkv` module internals and save `[B, heads, 576, 576]` tensors to disk (memory-mapped, one `.npy` per batch per layer)
2. Batch files consolidated into a per-layer `.npy` via memory-mapped concatenation
3. For each of the 8 layers: head-average → residual preservation (0.5A + 0.5I) → bottom-90% link pruning → row normalisation
4. Matrix-multiply rollout matrices across all 8 layers
5. **GAP adaptation**: global relevance = `rollout.sum(axis=1)` — sum over all query positions, not CLS-row extraction
6. Gini Coefficient and Shannon Entropy computed for all N patches, stratified by TP / TN / FP / FN
7. Rank-1 representative selected per group (closest to group median Gini) → visualised as 24 × 24 patch-grid heatmap

#### Integrated Gradients (optional)

Activated by `--compute-integrated-gradients`. Computes path-integrated gradients in **patch-token space**, not pixel space:

1. Zero baseline X₀
2. Gradients accumulated at `patch = to_patch_embedding(X) + pos_embedding` across 50 linearly spaced α steps
3. Average gradient multiplied by path delta `(patch_in − patch_base)`
4. Aggregated over the embedding dimension D via L2 norm → `[N, 576]` per-patch attribution map

A forward equivalence check verifies the decomposed forward path produces logits identical to `model(X)` (atol = rtol = 10⁻⁵) before any extraction begins.

**All outputs** saved to `{output-root}/{experiment}/analysis/`:

| File | Paper equivalent |
|------|-----------------|
| `embeddings_test_standard.npz` | — |
| `umap_comprehensive_test.png` | Figure 3 (4-panel) |
| `umap_slide_conditioned_test.png` | Supplementary |
| `umap_quality_metrics_test.json` | Section VI-A metrics |
| `rollout_stats_test.json` | Section VI-B statistics |
| `rollout_gini_distribution_test.png` | Section VI-B boxplot |
| `rollout_entropy_distribution_test.png` | Section VI-B boxplot |
| `rep_TP_XXXX_rollout_test.png` | Figure 4(a) |
| `rep_FP_XXXX_rollout_test.png` | Figure 4(b) |
| `rep_FN_XXXX_rollout_test.png` | Figure 4(c) |

---

## Configuration

All hyperparameters are centralised in `training/config.py` and committed as-is — identical to the version used in all paper experiments.

### BASE_CONFIG

| Parameter | Value | Description |
|-----------|-------|-------------|
| `random_seed` | 42 | Global base seed (SeedSequence root) |
| `image_size` | 384 | Input resolution (px) |
| `patch_size` | 16 | ViT patch size → 576 tokens |
| `dim` | 256 | Transformer embedding dimension |
| `depth` | 8 | Number of transformer encoder layers |
| `heads` | 6 | Number of attention heads |
| `mlp_dim` | 512 | Feed-forward network dimension |
| `channels` | 1 | Input channels (grayscale) |
| `n_iterations` | 30 | Independent CV iterations |
| `n_folds` | 5 | Folds per iteration |
| `num_epochs` | 100 | Maximum training epochs per fold |
| `patience` | 20 | Early stopping patience (epochs) |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 3 × 10⁻⁴ | AdamW base learning rate |
| `weight_decay` | 0.05 | AdamW weight decay |
| `warmup_epochs` | 5 | Linear LR warmup duration |
| `ema_alpha` | 0.3 | EMA smoothing factor for early stopping signal |
| `min_delta` | 0.01 | Minimum EMA improvement to reset patience |
| `gradient_clip_norm` | 1.0 | Gradient clipping max norm |
| `class_weight_gamma` | 0.0 | Class weight exponent (0 = uniform; recommended with MixUp) |
| `use_amp` | True | Automatic Mixed Precision (CUDA only) |
| `checkpoint_every_n_epochs` | 25 | Periodic checkpoint save frequency |

### STANDARD_CONFIG (augmentation, experiment = `standard`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `random_resized_crop_scale` | (0.9, 1.0) | RandomResizedCrop scale range |
| `random_resized_crop_ratio` | (0.95, 1.05) | RandomResizedCrop aspect ratio range |
| `horizontal_flip_p` | 0.5 | RandomHorizontalFlip probability |
| `vertical_flip_p` | 0.5 | RandomVerticalFlip probability |
| `rotation_degrees` | 30 | RandomRotation range (±30°) |
| `use_mixup` | True | Enable MixUp regularisation |
| `mixup_alpha` | 0.2 | MixUp Beta distribution parameter α |
| `mixup_prob` | 0.5 | Per-batch MixUp application probability |

---

## Key Design Decisions

Several methodological choices in this codebase are non-obvious but scientifically essential. This section documents them explicitly for reproducibility and critical review.

### 1 — Slide-Level Data Leakage Prevention

Every stage of the pipeline enforces that all images from a single biological slide are assigned exclusively to either training or test — never both. This prevents intra-slide correlation from inflating generalisation metrics, a well-documented pitfall in computational pathology (Bussola et al., 2020).

Enforced at every layer:

- `canonical_splits.py` — search operates on slides; images follow atomically
- `create_slide_level_folds()` — `StratifiedKFold` at slide level, mapped back to image indices; integrity validated by `validate_slide_grouping()`
- `calculate_dataset_stats()` — mean/std computed only from training-fold image indices
- `fit_basicpy_from_paths()` — BaSiCPy fitted only on training TIFFs identified by `split_indices.json`

### 2 — Two-Level Hierarchical Statistical Aggregation

Following Bengio & Grandvalet (2003): fold scores within each iteration are averaged first (collapsing within-iteration correlation), then statistics are computed across the 30 iteration means using t_{N−1} CIs. The alternative — treating all 150 folds as independent — overestimates statistical precision by √5 ≈ 2.24 and is scientifically incorrect.

### 3 — Lexicographic Model Selection

The best checkpoint per CV fold is selected by lexicographic ordering in `LexBestModelTracker`:

1. **Primary**: maximise slide-level Balanced Accuracy (clinically meaningful; unaffected by class imbalance)
2. **Tiebreaker**: minimise validation loss (resolves floating-point ties at ε = 10⁻⁶)
3. **Final tiebreaker**: prefer the earlier epoch

This criterion is deliberately decoupled from the early stopping signal (which operates on the EMA-smoothed BA) to prevent the stopping mechanism from biasing model selection.

### 4 — Attention Rollout for Global Average Pooling

Standard Attention Rollout (Abnar & Zuidema, 2020) extracts the CLS-token row from the final rollout matrix. SimpleViT uses Global Average Pooling instead — the classification signal is distributed across all 576 patch positions. The implementation in `extractor.py` adapts rollout by summing across all query positions (`rollout.sum(axis=1)`) to obtain a global relevance vector, which is then reshaped to the 24 × 24 patch grid for visualisation.

### 5 — Integrated Gradients in Patch-Token Space

Integrated Gradients are computed at the **patch-token level** rather than pixel space. The accumulation point is `patch = to_patch_embedding(X) + pos_embedding`. This is more principled for ViTs because it attributes at the granularity the transformer operates on, avoiding interpolation artefacts when mapping pixel-space gradients back onto a coarse patch grid. A forward equivalence check validates that the decomposed path produces identical logits before any extraction begins.

---

## Hardware and Compute

| Phase | Hardware | Approximate Duration |
|-------|----------|---------------------|
| Step 1 — Canonical split | CPU (any) | < 1 minute |
| Step 2 — Preprocessing (`optimized`) | CPU multi-core | 15–45 minutes |
| Step 3 — CV (30 × 5 folds) | NVIDIA A100 80GB (Colab Pro+) | 10–15 hours (resumable) |
| Step 3 — Final training | NVIDIA A100 80GB | 1–2 hours |
| Step 4 — CV statistics | CPU (any) | < 1 minute |
| Step 5 — UMAP + Rollout | CPU or GPU | 20–60 minutes |
| Step 5 — Integrated Gradients | GPU recommended | 1–3 hours |

Automatic Mixed Precision (`use_amp = True`) is enabled by default for CUDA, reducing VRAM usage by approximately 40% with no impact on metric reproducibility.

---

## Citation

If you use this code, dataset, or methodology in your research, please cite:

```bibtex
@inproceedings{sa2026explainable,
  title     = {Explainable Vision Transformers for {\em Candida} Species
               Identification from Brightfield Microscopy},
  author    = {Sá, Rodrigo and others},
  booktitle = {Proceedings of the IEEE World Congress on Computational
               Intelligence (WCCI)},
  year      = {2026},
  note      = {To appear}
}
```

*This entry will be updated with full bibliographic details (volume, pages, DOI) upon publication.*

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The dataset, when released, will be distributed under a separate open-access data licence to be specified at time of release.

---

## Acknowledgements

This work was conducted at [CISUC](https://www.cisuc.uc.pt/) — Centre for Informatics and Systems of the University of Coimbra — in collaboration with the Yeast Molecular Biology Lab (Nova University Lisbon).

The SimpleViT architecture is implemented via the [`vit-pytorch`](https://github.com/lucidrains/vit-pytorch) library (Wang, 2020). Illumination correction uses [`BaSiCPy`](https://github.com/peng-lab/BaSiCPy), a Python reimplementation of the BaSiC algorithm (Peng et al., 2017).

Generative AI tools (Gemini) and automated grammar checkers (Grammarly) were used solely for linguistic refinement and editorial assistance. All scientific content, experimental design, data analysis, and conclusions remain the full responsibility of the authors.

---

## References

- Abnar, S. & Zuidema, W. (2020). Quantifying attention flow in transformers. *Proceedings of the 58th Annual Meeting of the ACL*, 4190–4197.
- Bengio, Y. & Grandvalet, Y. (2003). No unbiased estimator of the variance of k-fold cross-validation. *Journal of Machine Learning Research, 5*, 1089–1105.
- Beyer, L., Zhai, X. & Kolesnikov, A. (2022). Better plain ViT baselines for ImageNet-1k. *arXiv:2205.01580*.
- Bussola, N. et al. (2020). AI slipping on tiles: data leakage in digital pathology. *arXiv:1909.06539*.
- Denning, D. W. (2024). Global incidence and mortality of severe fungal disease. *The Lancet Infectious Diseases, 24*(7), e428–e438.
- Dosovitskiy, A. et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR 2021*.
- Katsipoulaki, M. et al. (2024). *Candida albicans* and *Candida glabrata*: global priority pathogens. *Microbiology and Molecular Biology Reviews, 88*(2), e00021-23.
- Peng, T. et al. (2017). A BaSiC tool for background and shading correction of optical microscopy images. *Nature Communications, 8*, 14836.
- Sultana, S. et al. (2025). Microscopy-based fungi species classification using a deep learning ensemble approach. *ECCE 2025*.
- Sundararajan, M., Taly, A. & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML 2017*, 3319–3328.
- Zhang, H. et al. (2018). MixUp: Beyond empirical risk minimization. *ICLR 2018*.