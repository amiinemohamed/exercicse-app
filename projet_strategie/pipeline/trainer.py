import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.optimizers import OptimizerContext, OptimizerStrategy


class Trainer:
    """Responsable uniquement de l'entraînement du modèle.

    Le Trainer reçoit une stratégie, l'enveloppe dans un OptimizerContext
    et passe ce contexte au modèle. Changer d'algorithme d'optimisation =
    changer de stratégie, sans modifier ni le Trainer ni les modèles.
    """

    def __init__(self, model, strategy: OptimizerStrategy, epochs=500):
        self.model = model
        self.context = OptimizerContext(strategy)
        self.epochs = epochs

    def set_strategy(self, strategy: OptimizerStrategy):
        """Change d'algorithme d'optimisation à l'exécution."""
        self.context.set_strategy(strategy)

    def train(self, X, y):
        self.context.reset()
        # Appel du contexte : le modèle appellera context.update(...)
        self.model.fit(X, y, self.context, self.epochs)
        return self.model


if __name__ == "__main__":
    # Démonstration : même modèle, deux stratégies interchangeables.
    from core.dataset import Dataset
    from core.logisticRegression import LogisticRegression
    from core.optimizers import GradientDescent, MomentumGradientDescent
    from pipeline.evaluator import Evaluator

    root = Path(__file__).resolve().parent.parent
    X_train, y_train, X_test, y_test = Dataset(root / "data/patient_data.csv").split()

    trainer = Trainer(LogisticRegression(), GradientDescent(lr=0.1))
    for strategy in (GradientDescent(lr=0.1), MomentumGradientDescent(lr=0.1, beta=0.9)):
        trainer.set_strategy(strategy)
        model = trainer.train(X_train, y_train)
        print(f"{strategy!r:45} -> {Evaluator().evaluate(model, X_test, y_test)}")
