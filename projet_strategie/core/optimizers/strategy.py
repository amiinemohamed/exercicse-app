"""Interface Stratégie : tout algorithme d'optimisation doit la respecter."""

from abc import ABC, abstractmethod


class OptimizerStrategy(ABC):
    """Interface commune à toutes les stratégies d'optimisation.

    Une stratégie reçoit la liste des paramètres du modèle et la liste
    de leurs gradients, et met à jour les paramètres EN PLACE.
    """

    def __init__(self, lr=0.1):
        self.lr = lr

    @abstractmethod
    def update(self, params, grads):
        """Met à jour chaque paramètre à partir de son gradient."""

    def reset(self):
        """Réinitialise l'état interne (appelé avant chaque entraînement).

        Par défaut une stratégie n'a pas d'état ; les stratégies qui en
        ont un (ex. Momentum) redéfinissent cette méthode.
        """

    def __repr__(self):
        return f"{type(self).__name__}(lr={self.lr})"
