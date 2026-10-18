import numpy as np

from utils.preprocessing import compute_stats, standardize


class Model:
    """Classe de base : tout modèle doit avoir fit() et predict()."""

    def fit(self, X, y, optimizer, epochs=500):
        raise NotImplementedError

    def predict(self, X):
        """Retourne une probabilité d'infection entre 0 et 1 pour chaque patient."""
        raise NotImplementedError

    # --- outils communs aux modèles ---
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _fit_scaler(self, X):
        self.mean, self.std = compute_stats(X)

    def _scale(self, X):
        return standardize(np.asarray(X, dtype=float), self.mean, self.std)
