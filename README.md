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



---

## 💻 Tech Stack

### Machine Learning & Deep Learning
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TIMM](https://img.shields.io/badge/TIMM-0078D4?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### Frameworks & Tools
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

### Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 📊 Dataset & Model

### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| **Number of Classes** | 5 (Brown Rust, Healthy, Loose Smut, Septoria, Yellow Rust) |
| **Total Images** | Varies per dataset (balanced via GAN) |
| **Image Resolution** | 224×224×3 |
| **Data Split** | 80% train, 10% val, 10% test (3‑fold stratified) |
| **Augmentation** | Random flip, rotation, brightness‑contrast |

### Model Specifications

| Component | Details |
|-----------|---------|
| **CNN Backbone** | EfficientNet-B2 (pretrained on ImageNet) |
| **ViT Backbone** | ViT-Base-Patch16-224 (pretrained) |
| **BiLSTM** | Hidden size = 256, 2 layers, bidirectional, sequence length = 6 |
| **Attention** | Multi-Head Self-Attention (8 heads) |
| **Optimizer** | AdamW (lr = 3e-4, weight decay = 1e-4) |
| **Scheduler** | Cosine Annealing |
| **Loss** | Cross-Entropy with Label Smoothing |
| **GAN** | Conditional GAN, epochs = 20, feature matching |

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA‑enabled GPU (recommended for training)
- pip package manager

### Step-by-Step Setup

```bash
wheat-disease-detection/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 LICENSE
├── 📄 train.py                   # Main training script
├── 📄 inference.py               # Inference on single images
├── 📄 config.yaml                # Configuration file
├── 📁 models/
│   ├── 📄 __init__.py
│   ├── 📄 cnn_vit_lstm_attention.py   # Full model
│   ├── 📄 gan.py                     # Conditional GAN
│   └── 📄 utils.py
├── 📁 data/
│   ├── 📄 dataset.py                 # Dataset loading
│   ├── 📄 augmentation.py            # Data augmentation
│   └── 📄 preprocess.py
├── 📁 notebooks/
│   └── 📄 training_notebook.ipynb    # Jupyter notebook
├── 📁 results/
│   ├── 📄 confusion_matrices/
│   ├── 📄 roc_curves/
│   ├── 📄 grad_cam/
│   └── 📄 logs/
└── 📁 weights/
    ├── 📄 gan_generator.pth
    ├── 📄 classifier.pth
    └── 📄 scaler.pkl
```
## 📈 Performance Metrics

### 5‑Class Dataset Results

| Class | Precision | Recall | F1‑Score | Support |
|-------|-----------|--------|----------|---------|
| Brown Rust | 0.992 | 0.987 | 0.989 | 417 |
| Healthy | 0.998 | 0.996 | 0.997 | 535 |
| Loose Smut | 0.995 | 0.998 | 0.996 | 303 |
| Septoria | 0.986 | 0.981 | 0.983 | 119 |
| Yellow Rust | 0.991 | 0.989 | 0.990 | 465 |

| Metric | Value |
|--------|-------|
| **Accuracy** | **99.23%** |
| Macro F1 | 0.991 |
| Weighted F1 | 0.992 |

### 3‑Class Dataset Results

| Class | Precision | Recall | F1‑Score |
|-------|-----------|--------|----------|
| Brown Rust | 0.978 | 0.975 | 0.976 |
| Healthy | 0.998 | 0.996 | 0.997 |
| Yellow Rust | 0.972 | 0.979 | 0.975 |

| Metric | Value |
|--------|-------|
| **Accuracy** | **97.90%** |
| Macro F1 | 0.974 |
| Weighted F1 | 0.975 |

### Binary (2‑Class) Dataset Results

| Class | Precision | Recall | F1‑Score |
|-------|-----------|--------|----------|
| Diseased | 0.994 | 0.996 | 0.995 |
| Healthy | 0.998 | 0.997 | 0.998 |

| Metric | Value |
|--------|-------|
| **Accuracy** | **99.67%** |
| Macro F1 | 0.996 |
| Weighted F1 | 0.997 |

---

## 🔬 Ablation Study

| Model Variant | Accuracy | Key Observations |
|---------------|----------|------------------|
| **Full Model (Proposed)** | **99.23%** | Best performance with all components |
| Without GAN | 98.00% | Slight drop in minority class performance |
| Without LSTM & MHA | 96.12% | Reduced feature refinement and robustness |
| Without ViT | 95.05% | Poor discrimination between visually similar diseases |

**Takeaway:** Each component (ViT, BiLSTM, Attention, GAN) contributes significantly to the final performance, with synergistic effects enabling >99% accuracy.

---

## 📊 Results Visualization

### Confusion Matrices
Confusion matrices for the 5‑class dataset show strong diagonal dominance with minimal misclassifications between similar diseases (e.g., Yellow Rust vs. Brown Rust).

### ROC Curves
ROC curves for all classes achieve AUC > 0.99, confirming excellent discriminative ability.

### Grad‑CAM Heatmaps
Grad‑CAM visualizations highlight disease‑specific lesion areas (red/yellow) while ignoring healthy regions, proving the model learns biologically relevant features.

---

## 🌐 Deployment

### Local Deployment
```bash
# Run the Flask web app (optional)
python app.py


📞 Contact
Ayush Kumar
🎓 Roll No: 2308390100018
🏫 Department: Computer Science and Engineering
📧 Email: ayush.kumar@example.com
🔗 GitHub: @ayushkumar
💼 LinkedIn: Ayush Kumar
