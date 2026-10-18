import csv

import numpy as np


class Dataset:
    """Charge le fichier CSV des patients (dernière colonne : infected)."""

    def __init__(self, path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.feature_names = [k for k in rows[0] if k != "infected"]
        self.X = np.array([[float(r[k]) for k in self.feature_names] for r in rows])
        self.y = np.array([float(r["infected"]) for r in rows])

    def split(self, test_ratio=0.25, seed=0):
        idx = np.random.default_rng(seed).permutation(len(self.y))
        n_test = int(len(idx) * test_ratio)
        test, train = idx[:n_test], idx[n_test:]
        return self.X[train], self.y[train], self.X[test], self.y[test]
