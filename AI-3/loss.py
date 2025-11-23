import numpy as np


class BinaryCrossEntropy:
    def loss(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        self.y_true = y_true

        loss = - (y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred))
        return np.mean(loss)

    def grad(self):
        N = self.y_true.shape[1]
        grad = (-(self.y_true / self.y_pred) + (1 - self.y_true) / (1 - self.y_pred)) / N
        return grad


class CategoricalCrossEntropy:
    def loss(self, y_pred, y_true):

        m = y_true.shape[1]
        self.y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        self.y_true = y_true

        if y_true.ndim > 1 and y_true.shape[0] > 1:
            y_true = np.argmax(y_true, axis=0)

        log_likelihood = -np.log(self.y_pred[y_true, np.arange(m)])
        return np.mean(log_likelihood)

    def grad(self):
        m = self.y_true.shape[1]
        grad = self.y_pred.copy()

        if self.y_true.ndim > 1 and self.y_true.shape[0] > 1:
            y_true = np.argmax(self.y_true, axis=0)
        else:
            y_true = self.y_true.astype(int).flatten()

        grad[y_true, np.arange(m)] -= 1
        grad /= m
        return grad