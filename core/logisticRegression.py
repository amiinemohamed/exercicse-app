import numpy as np

from core.model import Model


class LogisticRegression(Model):
    def fit(self, X, y, optimizer, epochs=500):
        self._fit_scaler(X)
        X = self._scale(X)
        self.w = np.zeros(X.shape[1])
        self.b = np.zeros(1)
        for _ in range(epochs):
            err = self.sigmoid(X @ self.w + self.b) - y
            grad_w = X.T @ err / len(y)
            grad_b = np.array([err.mean()])
            optimizer.update([self.w, self.b], [grad_w, grad_b])

    def predict(self, X):
        return self.sigmoid(self._scale(X) @ self.w + self.b)
