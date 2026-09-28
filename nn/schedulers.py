"""
Learning rate schedulers.

Cada scheduler envolve um otimizador e modifica `optimizer.lr` a cada
época. Uso típico:

    opt = Adam(lr=1e-3)
    sched = CosineAnnealing(opt, t_max=20, eta_min=1e-5)

    for epoch in range(20):
        model.fit(..., epochs=1)
        sched.step()
        print(opt.lr)

Ou, integrado ao `fit`:
    model.fit(X, y, epochs=20, scheduler=sched)
"""

import numpy as np


class _LRScheduler:
    def __init__(self, optimizer, base_lr=None):
        self.optimizer = optimizer
        self.base_lr = base_lr if base_lr is not None else optimizer.lr
        self.last_epoch = 0

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()
        return self.optimizer.lr


class StepLR(_LRScheduler):
    """Multiplica o LR por `gamma` a cada `step_size` épocas."""

    def __init__(self, optimizer, step_size=10, gamma=0.1, base_lr=None):
        super().__init__(optimizer, base_lr)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return self.base_lr * factor


class ExponentialLR(_LRScheduler):
    """Multiplica o LR por `gamma` a cada época."""

    def __init__(self, optimizer, gamma=0.95, base_lr=None):
        super().__init__(optimizer, base_lr)
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** self.last_epoch)


class CosineAnnealing(_LRScheduler):
    """
    Cosine annealing (Loshchilov & Hutter, 2016).

        lr(t) = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * t / T))
    """

    def __init__(self, optimizer, t_max, eta_min=0.0, base_lr=None):
        super().__init__(optimizer, base_lr)
        self.t_max = t_max
        self.eta_min = eta_min

    def get_lr(self):
        t = min(self.last_epoch, self.t_max)
        cos = np.cos(np.pi * t / self.t_max)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + cos)


class WarmupCosine(_LRScheduler):
    """
    Warmup linear seguido de cosine annealing.

    Warmup ajuda em redes profundas nas primeiras épocas.
    """

    def __init__(self, optimizer, warmup_epochs, t_max, eta_min=0.0, base_lr=None):
        super().__init__(optimizer, base_lr)
        self.warmup_epochs = warmup_epochs
        self.t_max = t_max
        self.eta_min = eta_min

    def get_lr(self):
        t = self.last_epoch
        if t < self.warmup_epochs:
            return self.base_lr * (t + 1) / self.warmup_epochs
        progress = min(t - self.warmup_epochs, self.t_max) / self.t_max
        cos = np.cos(np.pi * progress)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + cos)
