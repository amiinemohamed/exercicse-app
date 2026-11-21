from utils.metrics import accuracy, precision, recall


class Evaluator:
    """Mesure la qualité d'un modèle entraîné sur des données de test."""

    def evaluate(self, model, X, y):
        y_pred = (model.predict(X) >= 0.5).astype(float)
        return {
            "accuracy": accuracy(y, y_pred),
            "precision": precision(y, y_pred),
            "recall": recall(y, y_pred),
        }
