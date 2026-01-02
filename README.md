# Earthquake Damage Detection (CNN vs Vision Transformers)

**Binary image classification:** `damaged` vs `undamaged` buildings after earthquakes.  
This project compares **CNN** and **Vision Transformer** backbones using a common training pipeline and reports results with **Accuracy, F1-score, and ROC-AUC**.  
Additionally, the best CNN and best Transformer are tested with a **SimCLR self-supervised pretraining → downstream evaluation** setup.

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Training Curves and Evaluation Plots](#training-curves-and-evaluation-plots)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)
- [Project Architecture](#project-architecture)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [Author Information](#author-information)

---

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd earthquake-damage-detection
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- `torch>=2.1` - PyTorch deep learning framework
- `torchvision>=0.16` - Computer vision utilities
- `timm>=0.9.12` - Pre-trained model library
- `numpy>=1.23` - Numerical computing
- `pandas>=2.0` - Data manipulation
- `scikit-learn>=1.3` - Machine learning utilities
- `Pillow>=10.0` - Image processing
- `pyyaml>=6.0` - YAML configuration parsing
- `tqdm>=4.66` - Progress bars
- `matplotlib>=3.7` - Plotting and visualization

---

## Project Structure

```
earthquake-damage-detection/
├── data/                          # Raw dataset
│   ├── damaged/                  # 1237 damaged building images
│   └── undamaged/                 # 1498 undamaged building images
│
├── outputs/                       # All experiment outputs
│   ├── splits/                    # Train/Val/Test CSV splits
│   ├── runs/                      # Model training results
│   │   ├── cnns/                  # CNN model outputs
│   │   ├── transformers/          # Vision Transformer outputs
│   │   └── simclr/                 # SimCLR model outputs
│   └── comparisons/                # Comparison tables and summaries
│
├── src/                           # Source code
│   ├── config/                    # YAML configuration files
│   │   ├── base.yaml              # Base configuration
│   │   ├── cnns.yaml              # CNN training config
│   │   ├── transformers.yaml     # Transformer training config
│   │   └── simclr.yaml            # SimCLR config
│   │
│   ├── data/                      # Data processing modules
│   │   ├── splits.py              # Dataset splitting
│   │   ├── dataset.py             # PyTorch Dataset class
│   │   ├── transforms.py          # Data augmentation
│   │   ├── simclr_dataset.py      # SimCLR-specific dataset
│   │   └── simclr_transforms.py   # SimCLR augmentations
│   │
│   ├── models/                     # Model definitions
│   │   ├── factory.py             # Model factory (timm-based)
│   │   └── simclr/                # SimCLR components
│   │       ├── encoder.py         # Feature encoder
│   │       ├── projector.py       # Projection head
│   │       └── loss.py            # NT-Xent loss
│   │
│   ├── train/                      # Training scripts
│   │   ├── train_cls.py           # Supervised classification
│   │   ├── train_simclr.py        # SimCLR pretraining
│   │   ├── finetune_from_simclr.py # SimCLR fine-tuning
│   │   ├── run_from_yaml.py       # YAML-based training
│   │   ├── run_simclr_from_yaml.py # SimCLR YAML runner
│   │   ├── compare_all.py         # Results comparison
│   │   └── export_test_preds.py  # Export predictions
│   │
│   ├── utils/                      # Utility functions
│   │   ├── metrics.py             # Evaluation metrics
│   │   ├── plots.py               # Plotting utilities
│   │   ├── seed.py               # Reproducibility
│   │   └── tables.py              # Table generation
│   │
│   └── viz/                       # Visualization scripts
│       ├── make_plots.py          # Supervised model plots
│       └── make_simclr_plots.py   # SimCLR model plots
│
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## Dataset

Images are organized under:
- `data/damaged` (**1237** images)
- `data/undamaged` (**1498** images)

**Total: 2735 images**

Train/Val/Test splits (seed=42) were generated and saved to `outputs/splits`:
- **Train:** 1914 (damaged=866, undamaged=1048)
- **Val:** 410 (damaged=185, undamaged=225)
- **Test:** 411 (damaged=186, undamaged=225)

Split ratio: 70% / 15% / 15% (stratified by class)

---

## Models

### Vision Transformers
- `vit_base_patch16_224` (ViT-Base, patch size 16)
- `swin_tiny_patch4_window7_224` (Swin-Tiny, patch size 4, window size 7)

### CNNs
- `resnet50` (ResNet-50)
- `efficientnet_b0` (EfficientNet-B0)
- `densenet121` (DenseNet-121)

All models are implemented using the `timm` library and trained with ImageNet pretrained weights (except SimCLR pretrained models).

---

## Evaluation Metrics

We report the following metrics on the **test split**:
- **Accuracy** - Overall classification accuracy
- **F1-score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under the ROC curve

All metrics are computed using `scikit-learn`.

---

## Results

### Summary Table (all runs)
The project automatically generates:
- `outputs/comparisons/summary_metrics.csv`
- `outputs/comparisons/summary_table.md`
- `outputs/comparisons/summary_table.png`

![Summary Table](outputs/comparisons/summary_table.png)

### Best Supervised Models
- **Best Transformer (supervised):**
  - `swin_tiny_patch4_window7_224`
  - **acc=0.9854, f1=0.9840, auc=0.9985**
- **Best CNN (supervised):**
  - `densenet121`
  - **acc=0.9830, f1=0.9814, auc=0.9992**

### SimCLR (Self-Supervised) Experiments
We applied **SimCLR pretraining** on the best CNN and best Transformer, then evaluated:
- **Linear evaluation** (freeze encoder + train linear head)
- **Fine-tuning** (train encoder + classifier head)

**Best SimCLR (linear):**
- `densenet121`  
- **acc=0.9805, f1=0.9785, auc=0.9991**

**Best SimCLR (fine-tune):**
- `densenet121`  
- **acc=0.9878, f1=0.9867, auc=0.9993**

✅ **Best overall model across all experiments:**  
- **SimCLR fine-tuned `densenet121`**
- **acc=0.9878, f1=0.9867, auc=0.9993**

---

## Training Curves and Evaluation Plots

### Supervised CNN Models

#### ResNet50
| Accuracy Curve | Loss Curve | F1 Curve | ROC Curve | Confusion Matrix |
|:---:|:---:|:---:|:---:|:---:|
| ![ResNet50 Accuracy](outputs/runs/cnns/resnet50_s42/plots/acc_curve.png) | ![ResNet50 Loss](outputs/runs/cnns/resnet50_s42/plots/loss_curve.png) | ![ResNet50 F1](outputs/runs/cnns/resnet50_s42/plots/f1_curve.png) | ![ResNet50 ROC](outputs/runs/cnns/resnet50_s42/plots/roc_curve.png) | ![ResNet50 CM](outputs/runs/cnns/resnet50_s42/plots/confusion_matrix.png) |

#### EfficientNet-B0
| Accuracy Curve | Loss Curve | F1 Curve | ROC Curve | Confusion Matrix |
|:---:|:---:|:---:|:---:|:---:|
| ![EfficientNet-B0 Accuracy](outputs/runs/cnns/efficientnet_b0_s42/plots/acc_curve.png) | ![EfficientNet-B0 Loss](outputs/runs/cnns/efficientnet_b0_s42/plots/loss_curve.png) | ![EfficientNet-B0 F1](outputs/runs/cnns/efficientnet_b0_s42/plots/f1_curve.png) | ![EfficientNet-B0 ROC](outputs/runs/cnns/efficientnet_b0_s42/plots/roc_curve.png) | ![EfficientNet-B0 CM](outputs/runs/cnns/efficientnet_b0_s42/plots/confusion_matrix.png) |

#### DenseNet121 (Best CNN)
| Accuracy Curve | Loss Curve | F1 Curve | ROC Curve | Confusion Matrix |
|:---:|:---:|:---:|:---:|:---:|
| ![DenseNet121 Accuracy](outputs/runs/cnns/densenet121_s42/plots/acc_curve.png) | ![DenseNet121 Loss](outputs/runs/cnns/densenet121_s42/plots/loss_curve.png) | ![DenseNet121 F1](outputs/runs/cnns/densenet121_s42/plots/f1_curve.png) | ![DenseNet121 ROC](outputs/runs/cnns/densenet121_s42/plots/roc_curve.png) | ![DenseNet121 CM](outputs/runs/cnns/densenet121_s42/plots/confusion_matrix.png) |

---

### Supervised Vision Transformer Models

#### ViT-Base
| Accuracy Curve | Loss Curve | F1 Curve | ROC Curve | Confusion Matrix |
|:---:|:---:|:---:|:---:|:---:|
| ![ViT-Base Accuracy](outputs/runs/transformers/vit_base_patch16_224_s42/plots/acc_curve.png) | ![ViT-Base Loss](outputs/runs/transformers/vit_base_patch16_224_s42/plots/loss_curve.png) | ![ViT-Base F1](outputs/runs/transformers/vit_base_patch16_224_s42/plots/f1_curve.png) | ![ViT-Base ROC](outputs/runs/transformers/vit_base_patch16_224_s42/plots/roc_curve.png) | ![ViT-Base CM](outputs/runs/transformers/vit_base_patch16_224_s42/plots/confusion_matrix.png) |

#### Swin-Tiny (Best Transformer)
| Accuracy Curve | Loss Curve | F1 Curve | ROC Curve | Confusion Matrix |
|:---:|:---:|:---:|:---:|:---:|
| ![Swin-Tiny Accuracy](outputs/runs/transformers/swin_tiny_patch4_window7_224_s42/plots/acc_curve.png) | ![Swin-Tiny Loss](outputs/runs/transformers/swin_tiny_patch4_window7_224_s42/plots/loss_curve.png) | ![Swin-Tiny F1](outputs/runs/transformers/swin_tiny_patch4_window7_224_s42/plots/f1_curve.png) | ![Swin-Tiny ROC](outputs/runs/transformers/swin_tiny_patch4_window7_224_s42/plots/roc_curve.png) | ![Swin-Tiny CM](outputs/runs/transformers/swin_tiny_patch4_window7_224_s42/plots/confusion_matrix.png) |

---

### SimCLR Self-Supervised Models

#### DenseNet121 with SimCLR (Best Overall Model)
| Accuracy Curve | Loss Curve | F1 Curve | ROC Curve | Confusion Matrix |
|:---:|:---:|:---:|:---:|:---:|
| ![DenseNet121 SimCLR Accuracy](outputs/runs/simclr/densenet121_s42/plots/acc_curve.png) | ![DenseNet121 SimCLR Loss](outputs/runs/simclr/densenet121_s42/plots/loss_curve.png) | ![DenseNet121 SimCLR F1](outputs/runs/simclr/densenet121_s42/plots/f1_curve.png) | ![DenseNet121 SimCLR ROC](outputs/runs/simclr/densenet121_s42/plots/roc_curve.png) | ![DenseNet121 SimCLR CM](outputs/runs/simclr/densenet121_s42/plots/confusion_matrix.png) |

#### Swin-Tiny with SimCLR
| Accuracy Curve | Loss Curve | F1 Curve | ROC Curve | Confusion Matrix |
|:---:|:---:|:---:|:---:|:---:|
| ![Swin-Tiny SimCLR Accuracy](outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/plots/acc_curve.png) | ![Swin-Tiny SimCLR Loss](outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/plots/loss_curve.png) | ![Swin-Tiny SimCLR F1](outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/plots/f1_curve.png) | ![Swin-Tiny SimCLR ROC](outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/plots/roc_curve.png) | ![Swin-Tiny SimCLR CM](outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/plots/confusion_matrix.png) |

---

## Key Findings

- In supervised training, **Swin-Tiny** produced the best Transformer results (**F1=0.9840**).
- Among CNNs, **DenseNet121** performed best in supervised mode (**F1=0.9814**).
- **SimCLR + fine-tuning improved DenseNet121** and produced the **best overall result** (**F1=0.9867**).
- For Swin-Tiny, SimCLR downstream performance did not surpass the supervised baseline in this dataset setting.

---

## How to Run

### 1) Create dataset splits (train / val / test)

This step scans the dataset directory, creates a dataset manifest, and generates train/validation/test splits using a fixed random seed.

**Command:**
```bash
python -m src.data.splits --seed 42
```

**Generated files:**
- `outputs/splits/train.csv`  
- `outputs/splits/val.csv`  
- `outputs/splits/test.csv`

These CSV files are used by all subsequent training steps.

---

### 2) Train supervised Vision Transformers

This step trains the Vision Transformer models (`vit_base_patch16_224` and `swin_tiny_patch4_window7_224`) using supervised learning.

**Command:**
```bash
python -m src.train.run_from_yaml --config src/config/transformers.yaml --seed 42
```

**Training outputs are saved to:**
- `outputs/runs/transformers/<model>_s42/`

Each run directory contains:
- `best.pt` - Best model checkpoint
- `history.csv` - Training history
- `metrics_test.json` - Test set metrics
- `plots/` - Training curves and evaluation plots
- `preds/` - Test predictions

---

### 3) Train supervised CNN models

This step trains the CNN-based models (`resnet50`, `efficientnet_b0`, and `densenet121`) using the same supervised pipeline.

**Command:**
```bash
python -m src.train.run_from_yaml --config src/config/cnns.yaml --seed 42
```

**Training outputs are saved to:**
- `outputs/runs/cnns/<model>_s42/`

---

### 4) Compare supervised runs (table + CSV + PNG)

This step aggregates all supervised CNN and Transformer results and generates comparison tables and figures.

**Command:**
```bash
python -m src.train.compare_all
```

**Generated files:**
- `outputs/comparisons/summary_metrics.csv`  
- `outputs/comparisons/summary_table.md`  
- `outputs/comparisons/summary_table.png`

---

## SimCLR Experiments

### 5) SimCLR pretraining (best CNN & best Transformer)

This step performs self-supervised SimCLR pretraining (without labels) on the best CNN and best Transformer models.

**Command:**
```bash
python -m src.train.run_simclr_from_yaml --config src/config/simclr.yaml --seed 42
```

**Generated SimCLR checkpoints:**
- `outputs/runs/simclr/densenet121_s42/simclr_pretrained.pt`  
- `outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/simclr_pretrained.pt`

---

### 6) Downstream evaluation from SimCLR checkpoints

#### DenseNet121

**Linear evaluation (encoder frozen)**  
The SimCLR-pretrained encoder is frozen and only a linear classifier is trained.

**Command:**
```bash
python -m src.train.finetune_from_simclr \
    --config src/config/simclr.yaml \
    --model densenet121 \
    --ckpt outputs/runs/simclr/densenet121_s42/simclr_pretrained.pt \
    --mode linear
```

**Fine-tuning (encoder + classifier head)**  
Both the encoder and the classification head are fine-tuned.

**Command:**
```bash
python -m src.train.finetune_from_simclr \
    --config src/config/simclr.yaml \
    --model densenet121 \
    --ckpt outputs/runs/simclr/densenet121_s42/simclr_pretrained.pt \
    --mode finetune
```

**Outputs are saved to:**
- `outputs/runs/simclr/densenet121_s42/linear/`  
- `outputs/runs/simclr/densenet121_s42/finetune/`

---

#### Swin-Tiny

**Linear evaluation (encoder frozen)**

**Command:**
```bash
python -m src.train.finetune_from_simclr \
    --config src/config/simclr.yaml \
    --model swin_tiny_patch4_window7_224 \
    --ckpt outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/simclr_pretrained.pt \
    --mode linear
```

**Fine-tuning (encoder + classifier head)**

**Command:**
```bash
python -m src.train.finetune_from_simclr \
    --config src/config/simclr.yaml \
    --model swin_tiny_patch4_window7_224 \
    --ckpt outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/simclr_pretrained.pt \
    --mode finetune
```

**Outputs are saved to:**
- `outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/linear/`  
- `outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/finetune/`

---

### 7) Generate plots for SimCLR models

After training SimCLR models, generate evaluation plots:

**Command:**
```bash
# For DenseNet121
python src/viz/make_simclr_plots.py --run_dir outputs/runs/simclr/densenet121_s42

# For Swin-Tiny
python src/viz/make_simclr_plots.py --run_dir outputs/runs/simclr/swin_tiny_patch4_window7_224_s42
```

**Note:** If `preds_test.npz` is missing, first export test predictions:
```bash
python src/train/export_test_preds.py \
    --run_dir outputs/runs/simclr/<model>_s42 \
    --split_csv outputs/splits/test.csv \
    --img_size 224
```

---

### 8) Regenerate final comparison table (includes SimCLR results)

This final step aggregates **all supervised and SimCLR-based experiments** and updates the final comparison table.

**Command:**
```bash
python -m src.train.compare_all
```

This step identifies:
- The best supervised CNN
- The best supervised Vision Transformer
- The best SimCLR (linear evaluation)
- The best SimCLR (fine-tuning)
- **The best overall model across all experiments**

---

## Project Architecture

### Configuration System

The project uses YAML-based configuration files for flexible hyperparameter management:

- **`src/config/base.yaml`** - Base dataset and split configuration
- **`src/config/cnns.yaml`** - CNN training hyperparameters
- **`src/config/transformers.yaml`** - Vision Transformer training hyperparameters
- **`src/config/simclr.yaml`** - SimCLR pretraining and fine-tuning configuration

### Data Pipeline

1. **Dataset Splitting** (`src/data/splits.py`)
   - Scans image directories
   - Creates stratified train/val/test splits
   - Saves CSV files with image paths and labels

2. **Data Loading** (`src/data/dataset.py`)
   - PyTorch `Dataset` class for CSV-based loading
   - Supports data augmentation via transforms

3. **Data Augmentation** (`src/data/transforms.py`)
   - Training: Random horizontal flip, color jitter, normalization
   - Validation/Test: Resize and normalization only

### Model Architecture

- **Supervised Models**: Built using `timm` library with ImageNet pretrained weights
- **SimCLR Models**: 
  - Encoder: Feature extractor (TimmEncoder)
  - Projector: MLP projection head (for pretraining)
  - Classifier: Linear head (for downstream tasks)

### Training Pipeline

1. **Supervised Training** (`src/train/train_cls.py`)
   - Standard classification training with early stopping
   - Saves best model based on validation metric (F1 or AUC)

2. **SimCLR Pretraining** (`src/train/train_simclr.py`)
   - Self-supervised contrastive learning
   - Uses NT-Xent loss with temperature scaling
   - Trains encoder + projector

3. **SimCLR Fine-tuning** (`src/train/finetune_from_simclr.py`)
   - Linear evaluation: Freeze encoder, train classifier
   - Fine-tuning: Train encoder + classifier with low learning rate

### Evaluation and Visualization

- **Metrics** (`src/utils/metrics.py`): Accuracy, F1, AUC, Confusion Matrix
- **Plots** (`src/utils/plots.py`, `src/viz/`): Training curves, ROC curves, confusion matrices
- **Comparison** (`src/train/compare_all.py`): Aggregates all results into comparison tables

---

## Outputs

All experiment artifacts are stored under the `outputs/` directory.

### Training Runs Structure

Each model run directory (`outputs/runs/{cnns|transformers|simclr}/<model>_s42/`) contains:

- **`best.pt`** - Best model checkpoint
- **`config_resolved.json`** - Resolved configuration used for training
- **`history.csv`** - Training history (epoch-by-epoch metrics)
- **`metrics_test.json`** - Final test set metrics
- **`plots/`** - Visualization directory:
  - `acc_curve.png` - Accuracy over epochs
  - `loss_curve.png` - Loss over epochs
  - `f1_curve.png` - F1 score over epochs
  - `roc_curve.png` - ROC curve with AUC
  - `confusion_matrix.png` - Confusion matrix
- **`preds/`** or **`preds_test.npz`** - Test predictions

### SimCLR Structure

SimCLR runs have additional structure:
- **`simclr_pretrained.pt`** - SimCLR pretrained encoder checkpoint
- **`simclr_history.csv`** - SimCLR pretraining history
- **`linear/`** - Linear evaluation results
- **`finetune/`** - Fine-tuning results

### Comparison Files

- **`outputs/comparisons/summary_metrics.csv`** - All metrics in CSV format
- **`outputs/comparisons/summary_table.md`** - Markdown comparison table
- **`outputs/comparisons/summary_table.png`** - Visual comparison table

**Note:** Only the comparison tables and figures are intended to be tracked in version control; large checkpoints and intermediate artifacts should remain local.

---

## Reproducibility

To reproduce the reported results:

1. **Use the same dataset structure** (`data/damaged`, `data/undamaged`)
2. **Generate splits with seed 42**:
   ```bash
   python -m src.data.splits --seed 42
   ```
3. **Run supervised training before SimCLR experiments**
4. **Use the same random seed (42)** for all experiments
5. **Regenerate the final comparison table** after all runs:
   ```bash
   python -m src.train.compare_all
   ```

### Reproducibility Features

- Fixed random seed (42) for all experiments
- Deterministic data splits (stratified)
- Consistent hyperparameters via YAML configs
- All configurations saved in `config_resolved.json`

---

## Limitations

- The dataset size is relatively small (2735 images); performance may differ on larger or more diverse datasets.
- SimCLR hyperparameters were not exhaustively tuned for each architecture.
- Only binary classification (damaged vs undamaged) is considered.
- Input image resolution is fixed at 224×224.
- Results may vary with different augmentation strategies or hyperparameters, especially for Transformer-based models.

---

## Conclusion

This study provides a systematic comparison between **CNNs** and **Vision Transformers** for earthquake damage detection.  
While Transformers perform strongly in supervised settings, **SimCLR self-supervised pretraining combined with fine-tuning yields the best overall performance with DenseNet121**, highlighting the benefits of representation learning for data-limited scenarios.

The project demonstrates:
- Effective use of modern deep learning architectures (CNNs and Vision Transformers)
- Application of self-supervised learning (SimCLR) for improved performance
- Comprehensive evaluation and comparison framework
- Reproducible experimental pipeline

---

## Author Information

**Name:** Emre Sebati Yolal  
**Student ID:** 244217003  
**Course:** YZ5511 – Derin Öğrenme ve Uygulamaları  

**Email:** emresebatiyolal@posta.mu.edu.tr  
**LinkedIn:** https://www.linkedin.com/in/esyolal/

---
