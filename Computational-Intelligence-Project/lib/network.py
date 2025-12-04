import numpy as np
from lib.layers import Dense
from lib.activations import ReLU, Sigmoid, Tanh, Softmax
from lib.losses import MeanSquaredError
from lib.optimizer import SGD

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss_function):
        self.optimizer = optimizer
        self.loss_function = loss_function

    def forward(self, X):
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backward(self, Y_pred, Y_true):
        grad = self.loss_function.gradient(Y_pred, Y_true)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_params(self):
        for layer in self.layers:
            layer.update_params(self.optimizer)

    def train(self, X, Y, epochs=1000):
        history = []
        for epoch in range(1, epochs+1):
            Y_pred = self.forward(X)
            loss = self.loss_function.loss(Y_pred, Y)
            history.append(loss)
            self.backward(Y_pred, Y)
            self.update_params()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}")
        return history

    def predict(self, X):
        return self.forward(X)
