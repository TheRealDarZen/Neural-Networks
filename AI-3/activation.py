import numpy as np

class Sigmoid:
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dY):
        return dY * self.y * (1 - self.y)


class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dY):
        dX = dY.copy()
        dX[self.x <= 0] = 0
        return dX


class Softmax:
    def forward(self, x):
        x_shifted = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.y = exp_x / np.sum(exp_x, axis=0, keepdims=True)
        return self.y

    def backward(self, dY):
        return dY