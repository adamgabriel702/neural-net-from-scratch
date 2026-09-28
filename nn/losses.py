import numpy as np


class Loss:
    """
    Funções de perda e suas derivadas em relação à ATIVAÇÃO DE SAÍDA (A).

    Convenção adotada:
      - A perda é `mean` sobre as amostras e `sum` sobre as dimensões de saída.
        Ou seja:  L = (1/m) * Σ_i L_i,  onde L_i agrega as saídas.
      - As derivadas retornadas por `*_backward` são "sum-reduced" sobre o batch
        (NÃO dividem por m). A divisão por m acontece em `Dense.backward`,
        aplicada somente em dW e db. Isso mantém o gradiente em relação às
        ativações intermediárias consistente ao longo do backpropagation.
    """

    # ---------------- MSE ----------------
    @staticmethod
    def mse(y_true, y_pred):
        y_true, y_pred = _as_2d(y_true), _as_2d(y_pred)
        return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))

    @staticmethod
    def mse_backward(y_true, y_pred):
        # dL/dA (sum-reduced): 2(A - y)
        return 2.0 * (y_pred - y_true)

    # ---------------- Binary Cross-Entropy ----------------
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        eps = 1e-15
        y_true, y_pred = _as_2d(y_true), _as_2d(y_pred)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        per_sample = np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred),
            axis=1,
        )
        return float(-np.mean(per_sample))

    @staticmethod
    def bce_backward(y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # dL/dA (sum-reduced) = (A - y) / (A(1 - A))
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    # ---------------- Categorical Cross-Entropy ----------------
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        eps = 1e-15
        y_true, y_pred = _as_2d(y_true), _as_2d(y_pred)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))

    @staticmethod
    def cce_backward(y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # dL/dA (sum-reduced) = -y / A
        return -y_true / y_pred


def _as_2d(a):
    a = np.asarray(a)
    return a.reshape(-1, 1) if a.ndim == 1 else a


LOSSES = {
    "mse": (Loss.mse, Loss.mse_backward),
    "bce": (Loss.binary_cross_entropy, Loss.bce_backward),
    "cce": (Loss.categorical_cross_entropy, Loss.cce_backward),
}