import numpy as np

class MeanSquaredError:
    @staticmethod
    def loss(Y_pred, Y_true):
        m = Y_pred.shape[1]
        return np.sum((Y_pred - Y_true)**2) / (2 * m)

    @staticmethod
    def gradient(Y_pred, Y_true):
        m = Y_pred.shape[1]
        return (Y_pred - Y_true) / m
