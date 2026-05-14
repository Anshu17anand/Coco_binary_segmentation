# COCO Binary Segmentation (U-Net, PyTorch)

This repository implements an end-to-end binary image segmentation pipeline on the COCO dataset using a U-Net model trained in PyTorch. The project focuses on correctness, experimental rigor, and clear evaluation under limited data and time constraints.

---

## Problem Setup

- **Task:** Binary segmentation (foreground vs background)
- **Dataset:** COCO 2017 (single category: dog)
- **Input:** RGB images resized to 256×256
- **Output:** Binary segmentation mask (1 = object, 0 = background)

The goal is to correctly classify each pixel as belonging to the target object or background.

---

## Model

- **Architecture:** U-Net (encoder–decoder with skip connections)
- **Why U-Net:** Designed for dense, pixel-level prediction and spatial localization
- **Framework:** PyTorch

---

## Dataset & Splits

- Total images used: ~250
- Train / Validation / Test split:
  - Train: 175
  - Validation: 37
  - Test: 38

Splits are created using image IDs to prevent data leakage.

---

## Training

- **Loss:** Binary Cross Entropy + Dice Loss
- **Optimizer:** Adam
- **Epochs:** 25
- **Batch size:** 4
- **Training:** From scratch (no pretrained encoder)

---

## Evaluation Metrics

- **Dice coefficient:** Measures overlap between predicted and ground-truth masks
- **IoU (Intersection over Union):** Derived from Dice for reporting

Final metrics:
- **Validation Dice:** ~0.24 (IoU ~0.14)
- **Test Dice:** ~0.21 (IoU ~0.12)

---

## Experiments & Findings

- Increasing dataset size improved generalization.
- Data augmentation (flip, crop, color jitter) was tested but reduced validation performance under the given training budget.
- Augmentation was removed based on empirical results.

This highlights the importance of evidence-driven experimentation rather than assuming best practices always apply.

---

## Failure Modes

Common failure cases include:
- Small or heavily occluded objects
- Low contrast scenes
- Truncated objects near image borders

---
<img width="1500" height="500" alt="image" src="https://github.com/user-attachments/assets/1d39f62b-2a66-44d7-a9e2-c36f979c73da" />


