# Build Your Own Neural Network Library & Advanced Applications

**Course:** MCT Program CSE473s: Computational Intelligence (Fall 2025)
**Project Type:** Python / NumPy Neural Network Implementation

---

## Project Overview

This project focuses on creating a foundational neural network library from scratch using **Python** and **NumPy**, without relying on high-level frameworks. The project is divided into three main parts:

1. **Library Implementation & Validation**
   Build a modular neural network library supporting layers, activations, losses, and optimizers. Validate the library by solving the classic **XOR problem**.

2. **Autoencoder & Latent Space Feature Extraction**
   Use the library to build an autoencoder for **MNIST image reconstruction**. Train the encoder and extract latent representations for further supervised classification using **SVM**.

3. **Comparative Analysis with TensorFlow/Keras**
   Re-implement the same networks in TensorFlow/Keras to compare ease of implementation, training time, and performance.

---

## Project Objectives

* **Deepen Understanding:** Implement forward and backward propagation and optimization from scratch.
* **Practical Implementation:** Create layers, activations, loss functions, and optimizers in Python.
* **Systematic Testing:** Use gradient checking to verify correctness of backpropagation.
* **Unsupervised Learning:** Implement autoencoders for dimensionality reduction and reconstruction.
* **Feature Extraction & Classification:** Train an SVM on latent features from the encoder.
* **Comparative Analysis:** Evaluate and compare performance against TensorFlow/Keras implementations.

---

## Core Library Requirements

The library must be modular and implemented **using only NumPy**.

### 1. Layers

* **Base Layer:** Abstract class with `forward()` and `backward()` methods.
* **Dense Layer:** Fully connected layer handling weights (`W`), biases (`b`), and gradients (`∂L/∂W`, `∂L/∂b`).

### 2. Activation Functions (Layer Subclasses)

* **ReLU:** `f(x) = max(0, x)`
* **Sigmoid:** `f(x) = 1 / (1 + e^(-x))`
* **Tanh:** `f(x) = (e^x - e^-x) / (e^x + e^-x)`
* **Softmax:** Normalized exponential for multi-class output.

### 3. Loss Functions

* **Mean Squared Error (MSE):** `L = (1/N) * Σ(Y_true - Y_pred)^2`

### 4. Optimizer

* **Stochastic Gradient Descent (SGD):** Updates weights using gradients and learning rate:
  `W_new = W_old - η * ∂L/∂W`

### 5. Network Model

* **Sequential / Network Class:** Orchestrates layers for full forward and backward passes and training loops.

---

## Project Structure

```
repo/
├── .gitignore
├── README.md
├── requirements.txt
├── lib/
│   ├── __init__.py
│   ├── layers.py
│   ├── activations.py
│   ├── losses.py
│   ├── optimizer.py
│   └── network.py
├── notebooks/
│   └── project_demo.ipynb
└── report/
    └── project_report.pdf
```

---

## Usage & Demonstration

### 1. Gradient Checking

Validate your backpropagation implementation using numerical gradient checking:

```
∂L/∂W ≈ [L(W + ε) - L(W - ε)] / (2ε)
```

### 2. XOR Problem

* **Architecture:** Example 2-4-1 MLP with Tanh/Sigmoid activations.
* **Training:** SGD with MSE loss.
* **Goal:** Correctly classify all four XOR inputs.

### 3. Autoencoder (MNIST)

* **Encoder:** Dense + ReLU layers to compress 784 pixels → latent space (32–64 dimensions).
* **Decoder:** Dense + ReLU/Sigmoid layers to reconstruct images.
* **Training:** Unsupervised using MSE loss.

### 4. Latent Space Classification

* Use trained encoder to extract features.
* Train an SVM classifier on latent features.
* Report metrics: test accuracy, confusion matrix, and classification evaluation.

### 5. TensorFlow/Keras Comparison

* Re-implement both XOR network and autoencoder.
* Compare:

  * Ease of implementation
  * Training time
  * Final reconstruction loss

---

## Deliverables

1. **Source Code** in `/lib`
2. **Jupyter Notebook** `/notebooks/project_demo.ipynb` demonstrating:

   * Gradient Checking
   * XOR Network training & results
   * Autoencoder training, loss curve, and reconstructed images
   * Latent Space SVM classification
   * TensorFlow/Keras implementations & comparison
3. **Project Report** `/report/project_report.pdf` covering:

   * Library design and architecture
   * XOR problem results
   * Autoencoder analysis
   * SVM classification analysis
   * TensorFlow comparison
   * Challenges and lessons learned


## Requirements

* Python 3.x
* NumPy
* Matplotlib
* scikit-learn (for SVM evaluation)
* Optional: TensorFlow/Keras for comparison

---

## References

* [NumPy Documentation](https://numpy.org/doc/)
* [TensorFlow Documentation](https://www.tensorflow.org/)
* CSE473s Course Material, Fall 2025

---

