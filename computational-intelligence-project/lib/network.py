import numpy as np
from lib.activations import Sigmoid, ReLU 
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input_data):
        raise NotImplementedError
    
    def backward(self, d_output):
        raise NotImplementedError
        
    def update_params(self, learning_rate):
        pass # Base layer has no parameters to update

# --- 2. Dense (Fully-Connected) Layer ---

class Dense(Layer):
    def __init__(self, input_size, output_size, activation_name="sigmoid"):
        super().__init__() 
        
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.b = np.zeros((1, output_size))
        
        # Store gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Select activation function (assuming ReLU and Sigmoid are imported)
        self.activation = ReLU if activation_name.lower() == "relu" else Sigmoid

        # Variables to store intermediate values for backprop
        self.input = None  # X
        self.A = None      # Activation(Z)

    def forward(self, input_data):
        """
        Performs the forward pass: Z = X * W + b, then A = activation(Z).
        """
        self.input = input_data
        
        # Linear part: Z = X @ W + b
        Z = np.dot(self.input, self.W) + self.b
        
        # Activation part: A = g(Z)
        self.A = self.activation.forward(Z)
        
        return self.A

    def backward(self, dA):
        """
        Calculates gradients dW, db, and dX (dL/dX).
        
        dA (np.ndarray): The gradient of the Loss with respect to the output (A).
        """
        m = self.input.shape[0] # Number of samples (batch size)
        
        # 1. Gradient of Z: dL/dZ = dL/dA * dA/dZ
        dZ = dA * self.activation.backward(self.A)

        # 2. Gradient of Weights: dL/dW = X_transpose * dL/dZ
        self.dW = np.dot(self.input.T, dZ) / m

        # 3. Gradient of Bias: dL/db = sum(dL/dZ, axis=0)
        self.db = np.sum(dZ, axis=0, keepdims=True) / m

        # 4. Gradient of Input: dL/dX = dL/dZ * W_transpose
        dX = np.dot(dZ, self.W.T)

        return dX
        
    def update_params(self, learning_rate):
        """ Applies the optimization rule (SGD) to update parameters. """
        # W = W - LR * dW
        self.W -= learning_rate * self.dW
        # b = b - LR * db
        self.b -= learning_rate * self.db


# --- 3. Neural Network Class (Unchanged for compatibility) ---

class NeuralNetwork:
    """
    Manages the overall network structure and training process.
    """
    def __init__(self, loss_func):
        # Now uses the list of Layer objects
        self.layers = []
        self.loss = loss_func
    
    def add_layer(self, layer):
        # Layer must be an instance of a class derived from Layer
        if not isinstance(layer, Layer):
             raise TypeError("Object must be an instance of a Layer subclass.")
        self.layers.append(layer)
        
    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self, Y_true, Y_pred):
        # 1. Start backprop by calculating the gradient of the Loss function
        dA = self.loss.backward(Y_true, Y_pred)
        
        # 2. Propagate the gradient backwards through all layers
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
            
    def update_params(self, learning_rate):
        """ Iterates through all layers and applies the update rule. """
        for layer in self.layers:
            # Calls the update_params method implemented in the Dense subclass
            layer.update_params(learning_rate)