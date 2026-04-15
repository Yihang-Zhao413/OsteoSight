# OsteoSight

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-informational)
![GUI](https://img.shields.io/badge/GUI-PyQt5-success)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

**OsteoSight: Label-Free Virtual Fluorescence Staining for Biophysics-Anchored Osteogenic Fate Inference from Conventional Microscopy**


</div>

---

## Overview

**OsteoSight** a biophysics-anchored computational system for robust, label-free osteogenic differentiation fate inference, designed to support:

- virtual fluorescence staining of subcellular structures and proteins from bright-field input
- biophysics-anchored feature extraction from VFS
- interpretable **osteogenic fate inference**
- interactive image acquisition and analysis through a desktop GUI

The current repository integrates four core components into a single workflow:

1. **Pre-step: Cellular image enhancement**
2. **Step 1: Virtual fluorescence staining**
3. **Step 2.1:Feature extraction and quantitative analysis**
4. **Step 2.2:Fate inference for cell osteogenic differentiation**

---

## Why OsteoSight?

Conventional differentiation assays often rely on endpoint staining, destructive measurements, or low-throughput biochemical readouts. OsteoSight aims to provide a more scalable alternative by enabling **non-invasive**, **image-based**, and **interpretable** analysis directly from label-free microscopy images.

It is especially useful for scenarios such as:

- longitudinal monitoring of osteogenic differentiation
- rapid screening of cell-state transitions
- extracting biologically meaningful image features
- building a deployable label-free imaging workflow for live-cell experiments

---

## Core Features

### 🔬 Label-free to virtual fluorescence staining
Generate virtual fluorescence-like predictions for key subcellular structures from bright-field input.

### 🧠 Interpretable fate inference
Move beyond a pure “image-to-label” black box by combining image-derived features with a dedicated stage prediction model.

### ✨ Two-phase image enhancement
A preprocessing pipeline combines **super-resolution / restoration** and **sparse deconvolution** to improve downstream virtual staining quality.

### 🎯 Multi-channel target support
Supports virtual staining for:

- **Nuclei**
- **F-actin**
- **YAP**

### 🖥️ Interactive desktop GUI
Includes a PyQt-based application for:

- camera control
- image capture / loading
- crop selection
- virtual staining
- ROI-based analysis
- fate prediction

### 📊 Quantitative feature extraction
Extracts morphology and biophysical descriptors, including:

- F1: YAP n/c ratio; F2: Alignment consistency; F3: Density; F4: Fractal dimension; F5: Eccentricity; F6: Elongation; F7: Radius max; F8: Radius min; F9: Perimeter; F10: Circularity; F11: Area.

---

## Workflow

```mermaid
flowchart LR
    A[Label-free bright-field image] --> B[Preprocessing<br/>SDRNet]
    B --> C[Virtual Staining<br/>WSCON]
    C --> D[ROI-based quantitative analysis]
    D --> E[Feature extraction]
    E --> F[Stage I / II / III inference]

---

## Citation and References

If you find this project useful, please also consider citing the following foundational works related to the methods used in this repository:


### References
- Park, T., Efros, A. A., Zhang, R. and Zhu, J.-Y. *Contrastive Learning for Unpaired Image-to-Image Translation*. ECCV, 2020.
- Isola, P., Zhu, J.-Y., Zhou, T. and Efros, A. A. *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR, 2017.
- Zhao, W., Zhao, S., Li, L. *et al.* *Sparse deconvolution improves the resolution of live-cell super-resolution fluorescence microscopy*. *Nature Biotechnology* 40, 606–617 (2022).
- Yue, Z., Wang, J. and Loy, C. C. *ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting*. NeurIPS, 2023.