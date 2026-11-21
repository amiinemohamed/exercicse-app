"""Contexte du pattern Stratégie."""

from core.optimizers.strategy import OptimizerStrategy


class OptimizerContext:
    """Détient une stratégie d'optimisation et lui délègue le travail.

    Les modèles ne parlent qu'au contexte (méthode update) : ils ignorent
    quel algorithme concret est utilisé, et on peut en changer à tout
    moment avec set_strategy() sans toucher aux modèles.
    """

    def __init__(self, strategy: OptimizerStrategy):
        self.set_strategy(strategy)

    @property
    def strategy(self) -> OptimizerStrategy:
        return self._strategy

    def set_strategy(self, strategy: OptimizerStrategy):
        if not isinstance(strategy, OptimizerStrategy):
            raise TypeError(
                f"{strategy!r} n'implémente pas l'interface OptimizerStrategy"
            )
        self._strategy = strategy

    def reset(self):
        self._strategy.reset()

    def update(self, params, grads):
        """Délégation à la stratégie courante."""
        self._strategy.update(params, grads)
