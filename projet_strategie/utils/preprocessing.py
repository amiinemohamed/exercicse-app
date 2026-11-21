def compute_stats(X):
    """Moyenne et écart-type de chaque colonne."""
    return X.mean(axis=0), X.std(axis=0) + 1e-8


def standardize(X, mean, std):
    """Centre et réduit les données."""
    return (X - mean) / std
