"""Stratégie concrète n°1 : descente de gradient simple."""

from core.optimizers.strategy import OptimizerStrategy


class GradientDescent(OptimizerStrategy):
    """p = p - lr * gradient"""

    def update(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g
