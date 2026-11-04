# Plant Disease Detection using PyTorch Custom CNN

This repository contains a professional PyTorch implementation of a Convolutional Neural Network (CNN) trained on the **PlantVillage** dataset to detect 38 different categories of crop diseases.

## Project Architecture
- **Dataset**: PlantVillage (43,444 train images, 10,861 test images)
- **Input Shape**: 3x128x128
- **Model**: Custom CNN (3 Conv Layers with Batch Normalization and Max Pooling + Fully Connected Classifier with Dropout)
- **Optimization**: Adam Optimizer (learning rate 0.001) with Cross-Entropy Loss

## Performance
- **Epochs**: 5
- **Final Test Accuracy**: ~92.23%

## File Structure
- `models/cnn_model.py`: Model Architecture
- `utils/dataset.py`: Custom PyTorch Dataset class
- `requirements.txt`: Python package dependencies
