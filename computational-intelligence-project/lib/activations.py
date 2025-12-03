import numpy as np

class ReLU:
    
    def forward(z):
        # Store the mask where input > 0 for use in the backward pass
       # ReLU.mask = (z > 0)
        return np.maximum(0, z)

    def backward(a):
        # Use the stored mask
        if a==0 or a<0 :
            a=0
        else :
            z=1
        return a

class Sigmoid:
    def forward(z):
        return 1 / (1 + np.exp(-z))

    def backward(a):
        return a * (1 - a)

class Tanh: 
    def forward(z):
        return np.tanh(z)

    def backward(a):
        return 1 - np.square(a)

class Softmax:
    
    def forward(z):
        max_z = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - max_z)    # (- max_z for overflow preventation 
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
        return exp_z / sum_exp_z

    def backward(a):
        """
        Calculates the derivative of Softmax. 
        Note: The derivative of Softmax is typically handled in combination with 
        Cross-Entropy Loss (Softmax-with-Loss) for simplicity and stability, 
        where dL/dZ = a - y_true. 
        
        If calculated standalone, it results in a complex Jacobian matrix. 
        For practical library building, we return a simple 1.0 placeholder
        or rely on the combined gradient if used with CrossEntropy.
        
        Since this method MUST return the local gradient (dA/dZ), and the 
        simple dL/dZ = a - y_true is a chain rule shortcut, we will use a 
        placeholder and rely on the combined gradient in the loss function
        for cross-entropy tasks later in the project.
        """
        # For simplicity in this modular structure, return 1 (dL/dZ=1 simplification)
        # However, a proper implementation would require a custom Softmax-Loss layer
        # For now, if used for multi-class, dZ is handled in the Loss class.
        return 1.0