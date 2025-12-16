import numpy as np
from lib.layers import Dense
from lib.activations import ReLU, Sigmoid, Tanh, Softmax
from lib.losses import MeanSquaredError
from lib.optimizer import GD

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

    def train(self, X, Y, epochs=1000, batch_size=None):
        """
        Trains the model using either Full-Batch or Mini-Batch Gradient Descent.

        Args:
            X: Input data (features, samples)
            Y: Target data (outputs, samples)
            epochs: Number of training iterations
            batch_size: Size of mini-batches. If None, uses Full-Batch GD.
        """
        history = []
        m = X.shape[1]  # Total number of samples

        for epoch in range(1, epochs + 1):

            # --- Option A: Full-Batch (Default) ---
            if batch_size is None:
                Y_pred = self.forward(X)
                loss = self.loss_function.loss(Y_pred, Y)
                self.backward(Y_pred, Y)
                self.update_params()

                epoch_loss = loss

            # --- Option B: Mini-Batch ---
            else:
                # 1. Shuffle data at the start of each epoch
                permutation = np.random.permutation(m)
                X_shuffled = X[:, permutation]
                Y_shuffled = Y[:, permutation]

                total_loss = 0

                # 2. Iterate over batches
                for i in range(0, m, batch_size):
                    # Slice batch (handles last batch automatically)
                    x_batch = X_shuffled[:, i : i + batch_size]
                    y_batch = Y_shuffled[:, i : i + batch_size]
                    current_batch_m = x_batch.shape[1]

                    # Forward & Backward for this batch
                    y_pred_batch = self.forward(x_batch)
                    batch_loss = self.loss_function.loss(y_pred_batch, y_batch)

                    self.backward(y_pred_batch, y_batch)
                    self.update_params()

                    # Accumulate weighted loss for accurate reporting
                    total_loss += batch_loss * current_batch_m

                epoch_loss = total_loss / m

            # --- Reporting ---
            history.append(epoch_loss)
            #if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}")

        return history

    def predict(self, X):
        return self.forward(X)