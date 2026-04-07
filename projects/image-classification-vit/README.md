# Image Classification (Vision Transformer)

## Overview

This project focuses on image classification using a pretrained Vision Transformer (ViT) model. The goal is to classify images into bird species and evaluate how model performance can be improved through preprocessing and training adjustments.

---

## My Contribution

My contribution focused on implementing and improving the baseline model.

I worked on:
- setting up the Vision Transformer (ViT) baseline
- preparing and preprocessing image data
- training and evaluating the model
- improving performance through extended fine-tuning

---

## Methods

The model used in this project is a pretrained Vision Transformer (ViT), which applies transformer architectures to image data.

Key steps:
- image preprocessing and normalization
- loading and fine-tuning a pretrained ViT model
- training on labeled image data
- evaluating model performance

---

## Results

The baseline model initially achieved an accuracy of approximately **0.48**.

After improving preprocessing and extending training, performance increased to approximately **0.74**.

This demonstrates the impact of proper training setup and preprocessing in deep learning models.

---

## Key Insight

Model performance in deep learning is not only determined by architecture, but also by training strategy and data preprocessing.

Even with a strong pretrained model, improvements in training setup can significantly increase performance.

---

## Repository Structure

```text
.
├── README.md
└── notebooks/
    └── baseline.ipynb
```
## Notes on Collaboration and Privacy

This project was developed as part of a university group project.

This repository contains only my individual contribution, specifically focused on the baseline model and preprocessing.

To respect collaboration boundaries, the following are not included:

- full group project files
- contributions from other team members
- datasets not owned by me
  
## Tech Stack
Python
PyTorch
Hugging Face Transformers
NumPy
