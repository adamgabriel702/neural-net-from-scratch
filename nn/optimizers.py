import numpy as np


class Optimizer:
    def step(self, layers):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            p, g = layer.params(), layer.grads()
            for k in p:
                p[k] -= self.lr * g[k]


class Momentum(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        self.lr, self.beta = lr, beta
        self.v = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            p, g = layer.params(), layer.grads()
            for k in p:
                key = (i, k)
                self.v.setdefault(key, np.zeros_like(p[k]))
                self.v[key] = self.beta * self.v[key] + (1 - self.beta) * g[k]
                p[k] -= self.lr * self.v[key]


class Adam(Optimizer):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            p, g = layer.params(), layer.grads()
            for k in p:
                key = (i, k)
                self.m.setdefault(key, np.zeros_like(p[k]))
                self.v.setdefault(key, np.zeros_like(p[k]))
                self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g[k]
                self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * (g[k] ** 2)
                m_hat = self.m[key] / (1 - self.b1 ** self.t)
                v_hat = self.v[key] / (1 - self.b2 ** self.t)
                p[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
