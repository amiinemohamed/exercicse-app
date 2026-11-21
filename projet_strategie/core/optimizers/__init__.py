"""Pattern Stratégie appliqué aux algorithmes d'optimisation."""

from core.optimizers.context import OptimizerContext
from core.optimizers.gradient_descent import GradientDescent
from core.optimizers.momentum import MomentumGradientDescent
from core.optimizers.strategy import OptimizerStrategy

__all__ = [
    "GradientDescent",
    "MomentumGradientDescent",
    "OptimizerContext",
    "OptimizerStrategy",
]
