import numpy as np


class BinaryCrossEntropy:
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        self.y_true = y_true

        loss = - (y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred))
        return np.mean(loss)

    def backward(self):
        N = self.y_true.shape[1]
        grad = (-(self.y_true / self.y_pred) + (1 - self.y_true) / (1 - self.y_pred)) / N
        return grad
