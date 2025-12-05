🧠 Computational Intelligence Project: Custom Neural Network Library

Project Title: Build Your Own Neural Network Library & Advanced Applications (Autoencoder & Latent Space SVM Classification)

This repository contains the deliverables for the CSE473s Computational Intelligence major project. The goal of this project was to develop a foundational, modular Neural Network library from scratch using only Python and NumPy, and subsequently demonstrate its power on complex applications like image reconstruction via an Autoencoder and supervised classification using latent space features.

🎯 Project Objectives

Deepen Understanding: Implement core machine learning algorithms (Forward/Backward Propagation, Optimization) from first principles.

Practical Implementation: Build modular components for layers, activations, loss functions, and optimizers.

Unsupervised Learning: Implement and train a custom Autoencoder on the MNIST dataset. 

Feature Extraction: Utilize the trained Encoder's latent space representation as features for an external SVM classifier.

Comparative Analysis: Benchmark the custom library's implementation and performance against the industry standard, TensorFlow/Keras.

📦 Repository Structure

The repository is organized according to the required submission structure:

.
├── lib/                             # Core Python Neural Network Library (NumPy only)
│   ├── __init__.py
│   ├── layers.py                    # Base Layer, Dense Layer
│   ├── activations.py               # ReLU, Sigmoid, Tanh, Softmax implementations
│   ├── losses.py                    # Mean Squared Error (MSE)
│   ├── optimizer.py                 # Stochastic Gradient Descent (SGD)
│   └── network.py                   # Sequential/Network class
├── notebooks/                       # Demonstrations and Experiments
│   └── project_demo.ipynb           # Comprehensive demo notebook (see sections below)
├── report/                          # Final Project Report
│   └── project_report.pdf           # Detailed analysis, design choices, and results
├── requirements.txt                 # Project dependencies (NumPy, Scikit-learn, TensorFlow, Matplotlib)
└── README.md                        # This file


⚙️ Getting Started

Prerequisites

To run the project locally, you need Python and the necessary libraries. It is highly recommended to use a virtual environment.

Clone the repository:

git clone [Your Repository URL]
cd [Your Repository Name]


Create and activate a virtual environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\activate   # On Windows


Install dependencies:

pip install -r requirements.txt


Running the Demo

The entire project demonstration, testing, and comparison are contained within the primary Jupyter Notebook.

Navigate to the notebooks/ directory.

Launch Jupyter Lab or Jupyter Notebook:

jupyter notebook


Open and run the cells in project_demo.ipynb.

📘 Project Demo Notebook Contents (/notebooks/project_demo.ipynb)

The notebook provides a step-by-step walkthrough of the library's functionality:

Section

Description

Key Deliverable

Section 1: Gradient Checking

Numerical vs. Analytical Gradient comparison to prove the correctness of the custom Backpropagation implementation.

$\ni L/\partial W\approx[L(W+\epsilon)-L(W-\epsilon)]/(2\epsilon)$

Section 2: XOR Problem

Validation of the custom library on the non-linear XOR problem using a simple MLP (e.g., 2-4-1).

Final predictions for all 4 XOR inputs.

Section 3: Custom Autoencoder

Training and visualization of the custom Autoencoder on the MNIST dataset using MSE loss.

Loss curve, original image vs. reconstructed image visualizations.

Section 4: Latent Space SVM

Using the trained Encoder to extract features and training an SVM classifier on the latent space representations.

Test Accuracy, Confusion Matrix, and Classification Report.

Section 5: TensorFlow Comparison

Implementation and training of the identical XOR and Autoencoder architectures using TensorFlow/Keras for direct comparison.

Comparison of Training Time, Implementation Ease, and Final Loss.

📝 Core Library Requirements (NumPy Only)

The lib/ directory implements the following components:

Layer Abstraction: Base Layer with forward() and backward().

Dense Layer: Handles weight initialization, forward pass, and gradient calculation for $W$ and $b$.

Activations: ReLU, Sigmoid, Tanh, and Softmax, each implemented with forward and backward (derivative) methods.

Losses: Mean Squared Error (MSE) implementation.

Optimizer: Stochastic Gradient Descent (SGD) for updating parameters: $W_{new}=W_{otd}-\eta(\partial L/\partial W)$.

Network Model: Sequential class to manage the layer list and orchestrate the training loop.