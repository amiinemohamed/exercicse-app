import numpy as np

from core.model import Model


class NeuralNetwork(Model):
    """Réseau à une couche cachée (tanh) et une sortie sigmoïde."""

    def __init__(self, hidden=8, seed=0):
        self.hidden = hidden
        self.seed = seed

    def fit(self, X, y, optimizer, epochs=500):
        self._fit_scaler(X)
        X = self._scale(X)
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, 0.5, (X.shape[1], self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, 0.5, self.hidden)
        self.b2 = np.zeros(1)
        for _ in range(epochs):
            h = np.tanh(X @ self.W1 + self.b1)
            p = self.sigmoid(h @ self.W2 + self.b2)
            err = (p - y) / len(y)
            dh = np.outer(err, self.W2) * (1 - h**2)
            optimizer.update(
                [self.W1, self.b1, self.W2, self.b2],
                [X.T @ dh, dh.sum(axis=0), h.T @ err, np.array([err.sum()])],
            )

    def predict(self, X):
        h = np.tanh(self._scale(X) @ self.W1 + self.b1)
        return self.sigmoid(h @ self.W2 + self.b2)
