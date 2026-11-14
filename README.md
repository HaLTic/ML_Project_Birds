# COSC 434 — Group 12: Bird Species Classification (CUB-200-2011)

**Authors:** Mohamed Alromaithi (`100060448`) & Zayed Alnuaimi (`100061300`)

## Overview
We compare conventional colour/texture classifiers against a frozen pretrained CNN on **CUB-200-2011** (`11,788` images, `200` bird species):
1. **RGB_LR** — 48-bin RGB histogram + Logistic Regression (baseline)
2. **HSV_SVM** — 512-bin joint HSV histogram + Linear SVM
3. **HSV_LBP_SVM** — HSV + 10-bin uniform Local Binary Pattern (LBP) + Linear SVM
4. **HSV_LBP_GLCM_SVM** — HSV + LBP + 12 Gray-Level Co-occurrence Matrix (GLCM) texture features + Linear SVM
5. **EfficientNet_LR** — Frozen ImageNet-pretrained `EfficientNet-B0` (1,280-D embeddings) + Logistic Regression

## Dataset & Fixed Splits
Download **CUB-200-2011** (`CUB_200_2011.tgz`, non-commercial research use) from the [official Caltech page](https://www.vision.caltech.edu/datasets/cub_200_2011/) and either place the archive in the project folder (the notebook extracts it on first run) or extract it yourself to `data/CUB_200_2011/`. If your copy lives elsewhere, edit `DATA_FOLDER` in Section 1 of the notebook. Raw images are excluded via `.gitignore`.

- **Split (`SEED = 42`):** Official test set (`5,794`) preserved untouched; official training set (`5,994`) stratified 80/20 into Training (`4,795`) and Validation (`1,199`).
- **Known Data Checks:** `8` grayscale (`L`) images are converted with `.convert('RGB')`; `1` exact duplicate pair (image IDs `3310` & `3340`, `058.Pigeon_Guillemot`) lies wholly inside Training (zero cross-split leakage).

## Quickstart
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Open [cosc434-group12-project-v2.ipynb](cosc434-group12-project-v2.ipynb) and **Run All** from a fresh kernel.
- **First run:** downloads the EfficientNet-B0 weights (about 20 MB) from the PyTorch servers, so it needs internet once. A GPU is used automatically if available but is not required.
- **Reruns:** extracted features are cached in `results_v2/features_*.npz` and `results_v2/cnn_features_*.npz`. Delete those files after changing any feature setting.
- **Outputs:** Saved to `results_v2/` (`test_metrics.csv`, `validation_trials.csv`, `top_confusion_pairs.csv`, confusion plots, and `settings.json`).

## Pretrained Resource Disclosure
- `EfficientNet_B0_Weights.IMAGENET1K_V1` (`torchvision`) is used strictly as a frozen feature extractor. As noted in the CUB-200-2011 documentation, ImageNet may overlap with some CUB photographs.
