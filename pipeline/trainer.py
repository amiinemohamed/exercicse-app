class Trainer:
    """Responsable uniquement de l'entraînement du modèle."""

    def __init__(self, model, optimizer, epochs=500):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs

    def train(self, X, y):
        self.model.fit(X, y, self.optimizer, self.epochs)
        return self.model
