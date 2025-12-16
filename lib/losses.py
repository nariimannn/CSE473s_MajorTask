import numpy as np

class MeanSquaredError:
    @staticmethod
    def loss(Y_pred, Y_true):
        # Ensure prediction and target shapes match
        assert Y_pred.shape == Y_true.shape, "Loss: prediction and target shapes must match"
        m = Y_pred.shape[1]  # batch size
        return np.sum((Y_pred - Y_true) ** 2) / (2 * m)

    @staticmethod
    def gradient(Y_pred, Y_true):
        assert Y_pred.shape == Y_true.shape, "Gradient: prediction and target shapes must match"
        m = Y_pred.shape[1]
        return (Y_pred - Y_true) / m

