# Earthquake Damage Detection (CNN vs Vision Transformers)
**Binary image classification:** `damaged` vs `undamaged` buildings after earthquakes.  
This project compares **CNN** and **Vision Transformer** backbones using a common training pipeline and reports results with **Accuracy, F1-score, and ROC-AUC**.  
Additionally, the best CNN and best Transformer are tested with a **SimCLR self-supervised pretraining → downstream evaluation** setup.

---

## Dataset
Images are organized under:
- `data/damaged` (**1237** images)
- `data/undamaged` (**1498** images)

Total: **2735** images

Train/Val/Test splits (seed=42) were generated and saved to `outputs/splits`:
- **Train:** 1914 (damaged=866, undamaged=1048)
- **Val:** 410 (damaged=185, undamaged=225)
- **Test:** 411 (damaged=186, undamaged=225)

---

## Models
### Vision Transformers
- `vit_base_patch16_224` (seed=42)
- `swin_tiny_patch4_window7_224` (seed=42)

### CNNs
- `resnet50` (seed=42)
- `efficientnet_b0` (seed=42)
- `densenet121` (seed=42)

---

## Evaluation Metrics
We report:
- **Accuracy**
- **F1-score**
- **ROC-AUC**

All results are computed on the **test split**.

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

## Key Findings
- In supervised training, **Swin-Tiny** produced the best Transformer results (**F1=0.9840**).
- Among CNNs, **DenseNet121** performed best in supervised mode (**F1=0.9814**).
- **SimCLR + fine-tuning improved DenseNet121** and produced the **best overall result** (**F1=0.9867**).
- For Swin-Tiny, SimCLR downstream performance did not surpass the supervised baseline in this dataset setting.

---

## How to Run

### 1) Create dataset splits (train / val / test)
This step scans the dataset directory, creates a dataset manifest, and generates train/validation/test splits using a fixed random seed.

Command:
- python -m src.data.make_splits --seed 42

Generated files are saved under:
- outputs/splits/train.csv  
- outputs/splits/val.csv  
- outputs/splits/test.csv  

These CSV files are used by all subsequent training steps.

---

### 2) Train supervised Vision Transformers
This step trains the Vision Transformer models (`vit_base_patch16_224` and `swin_tiny_patch4_window7_224`) using supervised learning.

Command:
- python -m src.train.run_from_yaml --config src/config/transformers.yaml --seed 42

Training outputs are saved to:
- outputs/runs/transformers/<model>_s42/

---

### 3) Train supervised CNN models
This step trains the CNN-based models (`resnet50`, `efficientnet_b0`, and `densenet121`) using the same supervised pipeline.

Command:
- python -m src.train.run_from_yaml --config src/config/cnns.yaml --seed 42

Training outputs are saved to:
- outputs/runs/cnns/<model>_s42/

---

### 4) Compare supervised runs (table + CSV + PNG)
This step aggregates all supervised CNN and Transformer results and generates comparison tables and figures.

Command:
- python -m src.train.compare_all

Generated files:
- outputs/comparisons/summary_metrics.csv  
- outputs/comparisons/summary_table.md  
- outputs/comparisons/summary_table.png  

---

## SimCLR Experiments

### 5) SimCLR pretraining (best CNN & best Transformer)
This step performs self-supervised SimCLR pretraining (without labels) on the best CNN and best Transformer models.

Command:
- python -m src.train.run_simclr_from_yaml --config src/config/simclr.yaml --seed 42

Generated SimCLR checkpoints:
- outputs/runs/simclr/densenet121_s42/simclr_pretrained.pt  
- outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/simclr_pretrained.pt  

---

### 6) Downstream evaluation from SimCLR checkpoints

#### DenseNet121

**Linear evaluation (encoder frozen)**  
The SimCLR-pretrained encoder is frozen and only a linear classifier is trained.

Command:
- python -m src.train.finetune_from_simclr --config src/config/simclr.yaml --model densenet121 --ckpt outputs/runs/simclr/densenet121_s42/simclr_pretrained.pt --mode linear

**Fine-tuning (encoder + classifier head)**  
Both the encoder and the classification head are fine-tuned.

Command:
- python -m src.train.finetune_from_simclr --config src/config/simclr.yaml --model densenet121 --ckpt outputs/runs/simclr/densenet121_s42/simclr_pretrained.pt --mode finetune

Outputs are saved to:
- outputs/runs/simclr/densenet121_s42/linear/  
- outputs/runs/simclr/densenet121_s42/finetune/  

---

#### Swin-Tiny

**Linear evaluation (encoder frozen)**  

Command:
- python -m src.train.finetune_from_simclr --config src/config/simclr.yaml --model swin_tiny_patch4_window7_224 --ckpt outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/simclr_pretrained.pt --mode linear

**Fine-tuning (encoder + classifier head)**  

Command:
- python -m src.train.finetune_from_simclr --config src/config/simclr.yaml --model swin_tiny_patch4_window7_224 --ckpt outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/simclr_pretrained.pt --mode finetune

Outputs are saved to:
- outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/linear/  
- outputs/runs/simclr/swin_tiny_patch4_window7_224_s42/finetune/  

---

### 7) Regenerate final comparison table (includes SimCLR results)
This final step aggregates **all supervised and SimCLR-based experiments** and updates the final comparison table.

Command:
- python -m src.train.compare_all

This step identifies:
- The best supervised CNN
- The best supervised Vision Transformer
- The best SimCLR (linear evaluation)
- The best SimCLR (fine-tuning)
- **The best overall model across all experiments**

## Outputs
All experiment artifacts are stored under the `outputs/` directory.

- **Training runs:**  
  - `outputs/runs/transformers/<model>_s42/`  
  - `outputs/runs/cnns/<model>_s42/`  
  - `outputs/runs/simclr/<model>_s42/{linear,finetune}/`

- **Final comparisons (recommended to commit):**  
  - `outputs/comparisons/summary_metrics.csv`  
  - `outputs/comparisons/summary_table.md`  
  - `outputs/comparisons/summary_table.png`

Only the comparison tables and figures are intended to be tracked in version control; large checkpoints and intermediate artifacts should remain local.

---

## Notes
- All experiments use a fixed random seed (**42**) for reproducibility.  
- Input image resolution is **224 × 224**.  
- Metrics are reported on the **test split** only.  
- SimCLR experiments include **linear evaluation** (encoder frozen) and **fine-tuning** (encoder + classifier).  
- Results may vary with different augmentation strategies or hyperparameters, especially for Transformer-based models.

---

## Reproducibility
To reproduce the reported results:
1. Use the same dataset structure (`data/damaged`, `data/undamaged`).
2. Generate splits with seed 42.
3. Run supervised training before SimCLR experiments.
4. Regenerate the final comparison table after all runs.

---

## Limitations
- The dataset size is relatively small; performance may differ on larger or more diverse datasets.
- SimCLR hyperparameters were not exhaustively tuned for each architecture.
- Only binary classification (damaged vs undamaged) is considered.

---

## Conclusion
This study provides a systematic comparison between **CNNs** and **Vision Transformers** for earthquake damage detection.  
While Transformers perform strongly in supervised settings, **SimCLR self-supervised pretraining combined with fine-tuning yields the best overall performance with DenseNet121**, highlighting the benefits of representation learning for data-limited scenarios.

---

## Author Information

**Name:** Emre Sebati Yolal  
**Student ID:** 244217003  
**Course:** YZ5511 – Derin Öğrenme ve Uygulamaları  

**Email:** emresebatiyolal@posta.mu.edu.tr  
**LinkedIn:** https://www.linkedin.com/in/esyolal/

---

