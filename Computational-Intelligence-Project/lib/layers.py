import numpy as np

class BaseLayer:
    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

    def update_params(self, optimizer):
        pass


class Dense(BaseLayer):
    def __init__(self, input_size, output_size, init_scale=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.A = np.random.randn(output_size, input_size) * init_scale
        self.b = np.zeros((output_size, 1))
        self.dA = None
        self.db = None
        self.f = None

    def forward(self, input_data):
        # input_data shape: (input_size, batch_size)
        self.f = input_data
        Z = np.dot(self.A, self.f) + self.b
        return Z

    def backward(self, dZ):
        # dZ shape: (output_size, batch_size)
        m = self.f.shape[1]  # batch size

        # Correct gradients
        self.dA = np.dot(dZ, self.f.T) / m  # shape: (output_size, input_size)
        self.db = np.sum(dZ, axis=1, keepdims=True) / m  # shape: (output_size, 1)

        # Gradient w.r.t input
        df_prev = np.dot(self.A.T, dZ)  # shape: (input_size, batch_size)
        return df_prev

    def update_params(self, optimizer):
        self.A = optimizer.step(self.A, self.dA)
        self.b = optimizer.step(self.b, self.db)

