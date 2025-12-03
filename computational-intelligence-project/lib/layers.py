
import numpy as np
from .activations import Sigmoid, ReLU, Tanh, Softmax 
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input_data):
        raise NotImplementedError
    
    def backward(self, d_output):
        raise NotImplementedError
    
    def update_params(self, learning_rate):
        pass 


class Dense(Layer):
    
    def __init__(self, input_size, output_size, activation_name="sigmoid"):
        super().__init__() 
        
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.b = np.zeros((1, output_size))
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        activations = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh, "softmax": Softmax}
        self.activation = activations.get(activation_name.lower(), Sigmoid)

        self.input = None  
        self.A = None      

    def forward(self, input_data):
        self.input = input_data
        Z = np.dot(self.input, self.W) + self.b
        self.A = self.activation.forward(Z)
        return self.A

    def backward(self, dA):
        m = self.input.shape[0] 
        dZ = dA * self.activation.backward(self.A)
        self.dW = np.dot(self.input.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = np.dot(dZ, self.W.T)
        return dX
    