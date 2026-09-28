import numpy as np


class Activation:
    """Ativações com forward e backward desacoplados (mais limpo que deriv)."""

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
        z = z - np.max(z, axis=1, keepdims=True)  # estabilidade numérica
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

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
    "softmax": (Activation.softmax, None),  # tratado junto com CE
    "linear":  (Activation.linear,  Activation.linear_backward),
}
