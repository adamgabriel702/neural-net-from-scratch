"""
Otimizadores e gradient clipping.

Todo otimizador suporta `clip_norm`, que aplica clip global de norma
l2 no gradiente ANTES de atualizar os parâmetros. Isso é importante
em RNNs, onde o BPTT pode produzir gradientes enormes (exploding
gradients) que destroem o treino.

Uso:
    opt = Adam(lr=1e-3, clip_norm=5.0)
    model.compile(loss="cce", optimizer=opt)
"""

import numpy as np


def clip_gradients(layers, max_norm):
    """
    Clipa globalmente a norma l2 de todos os gradientes concatenados.

    Se `norm > max_norm`, escala todos os gradientes por `max_norm / norm`.
    Modifica `layer.grads()[k]` in-place.

    Retorna a norma original (antes do clipping).
    """
    total_sq = 0.0
    for layer in layers:
        for g in layer.grads().values():
            if g is not None:
                total_sq += float(np.sum(g * g))

    total_norm = np.sqrt(total_sq)
    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / (total_norm + 1e-12)
        for layer in layers:
            for g in layer.grads().values():
                if g is not None:
                    g *= scale

    return total_norm


class Optimizer:
    """Classe base."""

    def __init__(self, lr=0.01, clip_norm=None):
        self.lr = lr
        self.clip_norm = clip_norm

    def _clip(self, layers):
        if self.clip_norm is not None:
            clip_gradients(layers, self.clip_norm)

    def step(self, layers):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.01, clip_norm=None):
        super().__init__(lr=lr, clip_norm=clip_norm)

    def step(self, layers):
        self._clip(layers)
        for layer in layers:
            p, g = layer.params(), layer.grads()
            for k in p:
                if g[k] is not None:
                    p[k] -= self.lr * g[k]


class Momentum(Optimizer):
    def __init__(self, lr=0.01, beta=0.9, clip_norm=None):
        super().__init__(lr=lr, clip_norm=clip_norm)
        self.beta = beta
        self.v = {}

    def step(self, layers):
        self._clip(layers)
        for i, layer in enumerate(layers):
            p, g = layer.params(), layer.grads()
            for k in p:
                if g[k] is None:
                    continue
                key = (i, k)
                self.v.setdefault(key, np.zeros_like(p[k]))
                self.v[key] = self.beta * self.v[key] + (1 - self.beta) * g[k]
                p[k] -= self.lr * self.v[key]


class Adam(Optimizer):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 clip_norm=None):
        super().__init__(lr=lr, clip_norm=clip_norm)
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, layers):
        self._clip(layers)
        self.t += 1
        for i, layer in enumerate(layers):
            p, g = layer.params(), layer.grads()
            for k in p:
                if g[k] is None:
                    continue
                key = (i, k)
                self.m.setdefault(key, np.zeros_like(p[k]))
                self.v.setdefault(key, np.zeros_like(p[k]))
                self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g[k]
                self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * (g[k] ** 2)
                m_hat = self.m[key] / (1 - self.b1 ** self.t)
                v_hat = self.v[key] / (1 - self.b2 ** self.t)
                p[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)