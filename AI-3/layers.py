import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim, std=0.01):
        self.W = np.random.randn(out_dim, in_dim) * std
        self.b = np.zeros((out_dim, 1))

    def forward(self, x):
        self.x = x
        return self.W @ x + self.b

    def backward(self, dY):
        self.dW = dY @ self.x.T
        self.db = np.sum(dY, axis=1, keepdims=True)
        return self.W.T @ dY

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
