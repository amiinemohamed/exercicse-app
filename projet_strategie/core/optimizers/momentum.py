"""Stratégie concrète n°2 : descente de gradient avec momentum."""

import numpy as np

from core.optimizers.strategy import OptimizerStrategy


class MomentumGradientDescent(OptimizerStrategy):
    """v = beta * v + gradient ;  p = p - lr * v

    Garde en mémoire une « vitesse » par paramètre, ce qui lisse les
    mises à jour et accélère la convergence.
    """

    def __init__(self, lr=0.1, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self._velocity = None

    def reset(self):
        self._velocity = None

    def update(self, params, grads):
        if self._velocity is None:
            self._velocity = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self._velocity):
            v *= self.beta
            v += g
            p -= self.lr * v

    def __repr__(self):
        return f"{type(self).__name__}(lr={self.lr}, beta={self.beta})"
