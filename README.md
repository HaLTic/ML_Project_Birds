# COSC 434 — Group 12: Bird Species Classification (CUB-200-2011)

**Authors:** Mohamed Alromaithi (`100060448`) & Zayed Alnuaimi (`100061300`)

## Overview
We compare hand-crafted colour/texture features against three frozen pretrained vision models on **CUB-200-2011** (`11,788` images, `200` bird species) in [cosc434-group12-project-v3-simple.ipynb](cosc434-group12-project-v3-simple.ipynb):

1. **Colour_LR** — 512-bin joint HSV colour histogram + Logistic Regression (baseline)
2. **ColourLBP_LR** — HSV + 10-bin uniform Local Binary Pattern (LBP) + Logistic Regression
3. **ColourLBPGLCM_LR** — HSV + LBP + 8 Gray-Level Co-occurrence Matrix (GLCM) features + Logistic Regression
4. **ColourLBPGLCM_SVM** — Same 530 colour/texture features + Linear SVM (`LinearSVC`, `dual=True`)
5. **ResNet18_LR** — Frozen ImageNet-pretrained `ResNet-18` (512-D) + Logistic Regression
6. **EfficientNet_LR** — Frozen ImageNet-pretrained `EfficientNet-B0` (1,280-D) + Logistic Regression
7. **DINOv2_LR** — Frozen self-supervised `DINOv2 ViT-S/14` (384-D) + Logistic Regression

## Dataset & Fixed Splits
Download **CUB-200-2011** (`CUB_200_2011.tgz`, non-commercial research use) from the [official Caltech page](https://www.vision.caltech.edu/datasets/cub_200_2011/) and place the archive in the project folder (extracted automatically on first run) or extract it to `data/CUB_200_2011/`. Raw images are excluded via `.gitignore`.

- **Splits (`SEED = 42`):** Official test set (`5,794`) preserved untouched; official training set (`5,994`) stratified 80/20 per species into Training (`4,795`) and Validation (`1,199`).
- **Known Data Notes:** `8` grayscale (`L`) images are converted with `.convert('RGB')`; `1` exact duplicate pair (image IDs `3310` & `3340`) lies wholly inside Training (zero cross-split leakage).

## Quickstart (Local or Kaggle)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Open [cosc434-group12-project-v3-simple.ipynb](cosc434-group12-project-v3-simple.ipynb) and **Run All** from a fresh kernel.
- **On Kaggle:** Attach `CUB-200-2011` as input (auto-detected under `/kaggle/input`) and enable Internet so pretrained weights can download on the first run. GPU accelerates feature extraction automatically.
- **Runtime, Hardware & Seed:** All splits and classifiers are locked with `SEED = 42`. Verified on Apple Silicon (`mps`) and Windows CPU; once features are cached (`results_v3_simple/*.npz`), all 28 validation tuning trials complete in `< 1 min` total (`results_v3_simple/validation_trials.csv`).
- **Reruns & Outputs:** Extracted feature caches (`results_v3_simple/*.npz`) are gitignored; final tables and plots (`test_metrics.csv`, `validation_trials.csv`, `top_confusion_pairs.csv`, `model_comparison.png`, `settings.json`) are saved in `results_v3_simple/`.

## Pretrained Resource Disclosure
- `ResNet18_Weights.IMAGENET1K_V1` and `EfficientNet_B0_Weights.IMAGENET1K_V1` (`torchvision`), plus `dinov2_vits14` (`facebookresearch/dinov2` via `torch.hub`), are used strictly as frozen feature extractors. As noted in the CUB-200-2011 documentation, pretraining datasets may overlap with some CUB photographs.
