
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, layers):
        if not isinstance(layers, list):
            layers = [layers]
            
        lr = self.learning_rate
        
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None:
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db