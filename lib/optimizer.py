class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, parameter, gradient):
        return parameter - self.learning_rate * gradient
