# Individual Machine Learning Project Proposal

**Student Name:** Nokutenda Prince Mahembe 
**Student ID:** 64215

### Project Title
Efficient Fashion Image Classification Using an Optimized Convolutional Neural Network

### 1. Problem Statement

Image classification models can require significant computational resources, making them difficult to run on devices with limited processing power. This project aims to develop a lightweight clothing classification system that maintains good prediction accuracy while reducing model size, inference time and computational requirements.

### 2. Project Objectives

- Develop a CNN capable of classifying clothing and accessories into ten categories.
- Apply structured pruning to reduce model size and computational requirements.
- Compare the performance of the original and optimized models.
- Develop a simple user interface to demonstrate the classification system.

### 3. Dataset

The Fashion-MNIST dataset will be used for this project. It contains 70,000 grayscale images with a resolution of 28 × 28 pixels, divided into ten clothing and accessory categories.

The dataset consists of 60,000 training images and 10,000 testing images.

### 4. Proposed Methodology

A baseline convolutional neural network will be developed and trained using PyTorch. Structured pruning will then be applied to remove less important filters and reduce the complexity of the model. The optimized model will be fine-tuned to maintain classification accuracy.

Both models will be evaluated and compared based on accuracy, parameter count, model size, inference latency and computational requirements.

A simple graphical user interface will be developed to allow users to select a test image, view its predicted clothing category, prediction confidence and inference time, and compare the baseline and optimized models.

### 5. Success Metrics

| Metric | Target |
|---|---|
| Classification accuracy | At least 90% |
| CPU inference latency | Below 50 ms per image |
| Model size | Below 5 MB |
| Parameter reduction | At least 40% |

### 6. Tools and Technologies

Python, PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Tkinter and Visual Studio 2022.

### 7. Expected Outcome

The expected outcome is a functional and efficient clothing classification system that can operate on a standard computer without requiring a dedicated GPU.

The project will demonstrate how model optimization can reduce computational costs while maintaining good classification performance. It will also provide a practical application of the machine learning and optimization techniques studied during the laboratory sessions. expect to have a working model that can recognize handwritten digits accurately. I will also demonstrate how model optimization can help make machine-learning systems more efficient.