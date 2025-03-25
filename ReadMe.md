# Quantum Convolutional Neural Network (QCNN) with MERA for Image Classification

## Overview

This project implements a hybrid quantum-classical neural network based on the paper:

**"Quantum convolutional neural network for image classification"**  
_Guoming Chen et al., Pattern Analysis and Applications, 2023_

The core idea is to use quantum circuits—specifically QCNNs built with MERA (Multi-scale Entanglement Renormalization Ansatz)—to extract and process multi-scale features from classical image datasets. This implementation focuses exclusively on MERA-based quantum preprocessing and evaluates it on a binary skin cancer classification task using the **ISIC 2018 dataset**.

---

## Project Highlights

- Implemented in **PennyLane** + **PyTorch**
- Uses MERA-based preprocessing pipelines:
  - **PCA** for dimensionality reduction
  - **HOG** (Histogram of Oriented Gradients) for edge-aware features
  - **PATCH-based encoding** to represent local image structures
- Quantum feature extraction is handled by **qml.templates.MERA**
- Hybrid model architecture using `qml.qnn.TorchLayer`
- Dataset: **ISIC 2018 (binary classification: benign vs malignant)**

---

## Preprocessing Methods

### 1. PCA + MERA

- Input image is resized and flattened.
- PCA is applied to extract the top `k` components (typically 16 or 32).
- Each component is scaled to fit the input range of RX, RY, RZ rotations.
- These values are then encoded into a fixed-depth MERA circuit.

### 2. HOG + MERA

- Histogram of Oriented Gradients is computed from grayscale input.
- The descriptor is flattened and truncated to the number of qubits used (e.g., 16).
- This forms the quantum feature vector input to the MERA circuit.

### 3. PATCH-based MERA Encoding

- The input image (e.g., 256×256) is divided into fixed-size patches (e.g., 16 patches of 64×64).
- Each patch is flattened and normalized.
- Each patch is encoded separately into a **16-qubit MERA** circuit.
- The outputs from all patches are concatenated to form a complete quantum feature vector.

---

## Quantum Circuit Design

### MERA Circuit (Quantum Feature Extractor)

- Based on `qml.templates.MERA`
- Hierarchically organized layers for multi-scale entanglement
- Uses `qml.qnn.TorchLayer` to embed it as the first layer of a PyTorch model
- Parameter counts are automatically determined based on:
  - `n_wires`: Number of qubits
  - `n_block_wires`: Size of the local block (set to 2 for entanglement)

### Hybrid Quantum-Classical Architecture

- Quantum layer outputs are treated as a fixed-size feature vector.
- This is passed into a fully connected classical neural network.
- Output: Sigmoid-activated scalar for binary classification (malignant vs benign).

---

## Dataset: ISIC 2018 Skin Cancer Classification

- Public dataset for skin lesion diagnosis
- This project elaborates the task to **multiclass classification**:
- Images are preprocessed using resizing, grayscale conversion, and contrast normalization.
- Label balancing and data augmentation (e.g., flipping) are optionally applied.

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

## Imp variables:

```bash
--n_qubits: Number of qubits (default: 16)

--patch_size: For patch-based encoding (default: 64x64)

--max_epochs: Number of training epochs

--batch_size: Mini-batch size
```
