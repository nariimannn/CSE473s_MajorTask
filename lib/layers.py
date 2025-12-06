import numpy as np

class BaseLayer:
    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

    def update_params(self, optimizer):
        pass


class Dense(BaseLayer):
    def __init__(self, input_size, output_size, init_scale=1.0):
        self.input_size = input_size
        self.output_size = output_size
        self.A = np.random.randn(output_size, input_size) * init_scale
        self.b = np.zeros((output_size, 1))
        self.dA = None
        self.db = None
        self.f = None   
        self.Z = None  

    def forward(self, input_data):
        # input_data shape: (input_size, batch_size)
        self.f = input_data
        self.Z = np.dot(self.A, self.f) + self.b
        return self.Z

    def backward(self, dZ):
        # dZ shape: (output_size, batch_size)
        m = self.f.shape[1]  # batch size

        self.dA = np.dot(dZ, self.f.T) / m   # (output_size, input_size)
        self.db = np.sum(dZ, axis=1, keepdims=True) / m  # (output_size, 1)

        # gradient w.r.t input to propagate
        dF_prev = np.dot(self.A.T, dZ)  # (input_size, batch_size)
        return dF_prev

    def update_params(self, optimizer):
        self.A = optimizer.step(self.A, self.dA)
        self.b = optimizer.step(self.b, self.db)
