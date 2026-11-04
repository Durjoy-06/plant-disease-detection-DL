# 🌿 Plant Disease Detection using Custom PyTorch CNN

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning repository designed to classify and detect plant leaf diseases from the **PlantVillage Dataset** using a custom Convolutional Neural Network (CNN) implemented in PyTorch. The model classifies leaf images into **38 distinct categories** covering crops like Apple, Grape, Tomato, Potato, Corn, and more.

## 📊 Performance Summary
- **Classification Test Accuracy**: **92.23%** (Achieved in just 5 epochs of training)
- **Dataset Size**: 54,305 total images (43,444 Train | 10,861 Test)
- **Framework**: PyTorch 2.x with CUDA Acceleration

---

## 📂 Repository Structure

```directory
├── models/
│   └── cnn_model.py      # Custom CNN (MyCNN) Architecture
├── utils/
│   └── dataset.py        # PyTorch Custom Dataset class for PlantVillage
├── predict.py            # CLI Script for quick inference/prediction on new images
├── requirements.txt      # Python environment dependencies
├── README.md             # Project documentation
└── plant_disease_model.pth # Saved weights of the trained model (92.23% Acc)
```

---

## 🛠️ Model Architecture

The model utilizes a custom designed **Deep Convolutional Neural Network (MyCNN)** optimized for fast computation and high accuracy on crop leaves:

1. **Feature Extractor**:
   - **Conv Block 1**: `Conv2d(3 -> 32, 3x3)` + `ReLU` + `BatchNorm2d` + `MaxPool2d(2x2)`
   - **Conv Block 2**: `Conv2d(32 -> 64, 3x3)` + `ReLU` + `BatchNorm2d` + `MaxPool2d(2x2)`
   - **Conv Block 3**: `Conv2d(64 -> 128, 3x3)` + `ReLU` + `BatchNorm2d` + `MaxPool2d(2x2)`

2. **Classifier Block**:
   - `Flatten` (128 x 16 x 16 -> 32,768)
   - `Linear(32768 -> 128)` + `ReLU` + `Dropout(p=0.4)`
   - `Linear(128 -> 64)` + `ReLU` + `Dropout(p=0.4)`
   - `Linear(64 -> 38)` (Softmax-ready logits for 38 disease classes)

---

## ⚙️ Hyperparameters

| Parameter | Value |
| :--- | :--- |
| **Input Image Size** | 128 x 128 (RGB) |
| **Batch Size** | 64 |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Loss Function** | Cross-Entropy Loss |
| **Epochs** | 5 |

---

## 🚀 How to Setup and Run

### 1. Clone the Repository
```bash
git clone https://github.com/Durjoy-06/plant-disease-detection-DL.git
cd plant-disease-detection-DL
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Quick Inference (Predicting leaf disease on an image)
You can run prediction on any local image file using our inference CLI script:
```bash
python predict.py path/to/your/leaf_image.jpg
```

---

## 📈 Experimental Results

- **Loss Reduction**: Smooth convergence from an initial Cross-Entropy loss of **1.8582** in Epoch 1 down to **0.5980** in Epoch 5.
- **Accuracy Progression**:
  - **Epoch 1**: 73.23%
  - **Epoch 2**: 82.13%
  - **Epoch 3**: 86.84%
  - **Epoch 4**: 89.37%
  - **Epoch 5**: **92.23%**

---

## 📜 License
Distributed under the MIT License. See `LICENSE` (or standard MIT terms) for more information.
