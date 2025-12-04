import numpy as np
from lib.layers import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        self.input_Z = None

    def forward(self, input_Z):
        self.input_Z = input_Z
        return np.maximum(0, input_Z)

    def backward(self, dA):
        return dA * (self.input_Z > 0).astype(float)


class Sigmoid(BaseLayer):
    def __init__(self):
        self.output_A = None

    def forward(self, input_Z):
        self.output_A = 1 / (1 + np.exp(-input_Z))
        return self.output_A

    def backward(self, dA):
        return dA * self.output_A * (1 - self.output_A)


class Tanh(BaseLayer):
    def __init__(self):
        self.output_A = None

    def forward(self, input_Z):
        self.output_A = np.tanh(input_Z)
        return self.output_A

    def backward(self, dA):
        return dA * (1 - self.output_A**2)


class Softmax(BaseLayer):
    def __init__(self):
        self.output_A = None

    def forward(self, input_Z):
        exps = np.exp(input_Z - np.max(input_Z, axis=0, keepdims=True))
        self.output_A = exps / np.sum(exps, axis=0, keepdims=True)
        return self.output_A

    def backward(self, dA):
        return dA
