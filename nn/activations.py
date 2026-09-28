import numpy as np


class Activation:
    """
    Funções de ativação com forward e backward desacoplados.

    Convenção de eixo: operações que agem "sobre as classes" (softmax)
    usam `axis=-1`. Isso funciona uniformemente para tensores 2D `(N, C)`,
    3D `(N, T, C)` e 4D `(N, H, W, C)`.
    """

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_backward(z, a):
        return a * (1 - a)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_backward(z, a):
        return (z > 0).astype(z.dtype)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_backward(z, a):
        return 1 - a ** 2

    @staticmethod
    def softmax(z):
        # `axis=-1` para suportar (N, C), (N, T, C) e (N, H, W, C).
        # `axis=1` seria o eixo do tempo em tensores 3D — errado.
        z = z - np.max(z, axis=-1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    @staticmethod
    def linear(z):
        return z

    @staticmethod
    def linear_backward(z, a):
        return np.ones_like(z)


ACTIVATIONS = {
    "sigmoid": (Activation.sigmoid, Activation.sigmoid_backward),
    "relu":    (Activation.relu,    Activation.relu_backward),
    "tanh":    (Activation.tanh,    Activation.tanh_backward),
    "softmax": (Activation.softmax, None),  # tratado junto com CCE em Sequential
    "linear":  (Activation.linear,  Activation.linear_backward),
}