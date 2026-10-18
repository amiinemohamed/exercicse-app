class GradientDescent:
    """Descente de gradient simple : p = p - lr * gradient."""

    def __init__(self, lr=0.1):
        self.lr = lr

    def update(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g
