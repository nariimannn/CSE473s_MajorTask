import numpy as np

# --- Loss Functions ---

class MeanSquaredError:
    """The Mean Squared Error (MSE) loss function."""
    
    @staticmethod
    def forward(y_true, y_pred):
        """
        Calculates the MSE loss: (1/m) * sum((y_pred - y_true)^2)
        """
        return np.mean(np.square(y_pred - y_true))

    @staticmethod
    def backward(y_true, y_pred):
        """
        Calculates the gradient of the MSE loss with respect to the prediction (y_pred).
        d Loss / d y_pred = (2/m) * (y_pred - y_true)
        """
        m = y_true.shape[0] # Number of samples
        # This is the initial 'dA' passed to the final layer's backward method
        return 2 * (y_pred - y_true) / m

class BinaryCrossEntropy:
    """Binary Cross-Entropy (BCE) loss function (ideal for the XOR problem)."""
    
    @staticmethod
    def forward(y_true, y_pred):
        """
        Calculates the BCE loss: - (1/m) * sum(y_true*log(y_pred) + (1-y_true)*log(1-y_pred))
        """
        # Clip y_pred to avoid log(0) or log(1-0) which are mathematically undefined
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def backward(y_true, y_pred):
        """
        Calculates the gradient of the BCE loss with respect to the prediction (y_pred).
        """
        # Clip y_pred to avoid division by zero
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        m = y_true.shape[0]
        # dL/dY_pred
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / m
