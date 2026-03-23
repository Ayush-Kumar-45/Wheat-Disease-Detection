# 🌾 GAN-Augmented Hybrid CNN-Vision Transformer-BiLSTM Framework with Attention for Wheat Disease Detection and Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)
![Vision Transformer](https://img.shields.io/badge/ViT-Base--16--224-orange)
![BiLSTM](https://img.shields.io/badge/BiLSTM-256%20units-success)
![Accuracy](https://img.shields.io/badge/5--Class%20Accuracy-99.23%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Paper](https://img.shields.io/badge/Paper-Accepted-important)

**A state-of-the-art deep learning framework for accurate and interpretable wheat leaf disease classification, combining GAN-based data augmentation, CNN-ViT dual-path feature extraction, BiLSTM temporal modeling, and multi-head attention.**

[📄 Paper](#) • 
[📊 Results](#performance-metrics) • 
[🛠️ Installation](#installation) • 
[🚀 Quick Start](#usage-guide)

</div>

---

## 📋 Table of Contents

- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Dataset & Model](#dataset--model)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Ablation Study](#ablation-study)
- [Results Visualization](#results-visualization)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 🎯 About The Project

Wheat is a staple food for billions, but its production is severely threatened by diseases like rust, smut, and septoria. Early and accurate detection is crucial for food security. However, traditional methods suffer from class imbalance, domain gaps, and limited generalizability.

This project presents a **GAN-Augmented Hybrid Deep Learning Framework** that:
- **Generates synthetic samples** for underrepresented disease classes using a conditional GAN.
- **Extracts both local and global features** via a dual-path CNN + Vision Transformer architecture.
- **Models sequential dependencies** with a Bidirectional LSTM on feature-level sequences.
- **Refines discriminative features** using a multi-head attention mechanism.

The result is a robust, highly accurate system validated on multiple datasets with 99.23% accuracy on a 5-class benchmark.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **Conditional GAN** | Generates realistic disease‑specific images to mitigate class imbalance. |
| 🔍 **Dual‑Path Extraction** | Combines CNN (EfficientNet‑B2) local textures with Vision Transformer global context. |
| ⏳ **BiLSTM Modeling** | Learns feature‑level temporal dependencies even from static images. |
| 🎯 **Multi‑Head Attention** | Dynamically focuses on the most discriminative disease patterns. |
| 📊 **Multi‑Dataset Validation** | Tested on 2‑class, 3‑class, and 5‑class datasets with 3‑fold cross‑validation. |
| 🖼️ **Grad‑CAM Interpretability** | Visual heatmaps highlight lesion regions, ensuring decisions are biologically relevant. |
| ⚡ **End‑to‑End Training** | Single, trainable pipeline from raw images to classification. |
| 🌾 **Precision Agriculture Ready** | Designed for real‑world field conditions with diverse backgrounds. |

---

## 🏗 System Architecture
