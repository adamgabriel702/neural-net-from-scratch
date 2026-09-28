"""
Camadas recorrentes: SimpleRNN e LSTM.

Convenção:
- Entrada:  X (N, T, D) — batch, timesteps, features
- Saída:    (N, H)      se `return_sequences=False`
            (N, T, H)   se `return_sequences=True`

Backpropagation através do tempo (BPTT) é implementada à mão.
A convenção de escala segue o resto do pacote: gradientes dos parâmetros
são divididos por N (batch size); o gradiente da entrada (`dX`) fica
per-sample para não ser dividido duas vezes ao longo da cadeia.
"""

import numpy as np

from .layers import Layer
from .activations import Activation


def _sigmoid(z):
    return Activation.sigmoid(z)


# ============================================================
# SimpleRNN
# ============================================================
class SimpleRNN(Layer):
    """
    RNN vanilla:

        h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b)

    Parâmetros
    ----------
    input_dim : int
    hidden_dim : int
    return_sequences : bool
        Se True, devolve toda a sequência de hidden states.
        Se False, devolve apenas o último.
    init : {"xavier", "he", "small"}
    seed : int ou None
    """

    def __init__(self, input_dim, hidden_dim, return_sequences=False,
                 init="xavier", seed=None):
        rng = np.random.default_rng(seed)
        if init == "he":
            scale = np.sqrt(2.0 / input_dim)
        elif init == "xavier":
            scale = np.sqrt(1.0 / input_dim)
        else:
            scale = 0.01

        self.Wx = (rng.standard_normal((input_dim, hidden_dim)) * scale).astype(np.float64)
        self.Wh = (rng.standard_normal((hidden_dim, hidden_dim)) * scale).astype(np.float64)
        self.b = np.zeros((1, hidden_dim), dtype=np.float64)

        self.dWx = None
        self.dWh = None
        self.db = None

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        self._cache = None

    def forward(self, X, training=True):
        assert X.ndim == 3, f"SimpleRNN espera (N, T, D), recebeu shape {X.shape}"
        N, T, D = X.shape
        assert D == self.input_dim

        h = np.zeros((N, self.hidden_dim))
        H = np.zeros((N, T, self.hidden_dim))

        for t in range(T):
            z = X[:, t, :] @ self.Wx + h @ self.Wh + self.b
            h = np.tanh(z)
            H[:, t, :] = h

        self._cache = (X, H)
        return H if self.return_sequences else H[:, -1, :]

    def backward(self, dout):
        X, H = self._cache
        N, T, D = X.shape
        Hd = self.hidden_dim

        if self.return_sequences:
            dH = dout
        else:
            dH = np.zeros((N, T, Hd))
            dH[:, -1, :] = dout

        dX = np.zeros_like(X)
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)

        dh_next = np.zeros((N, Hd))

        for t in range(T - 1, -1, -1):
            dh = dH[:, t, :] + dh_next
            h_t = H[:, t, :]

            # tanh'(z) = 1 - tanh(z)^2
            dz = dh * (1 - h_t ** 2)

            self.dWx += X[:, t, :].T @ dz
            h_prev = H[:, t - 1, :] if t > 0 else np.zeros((N, Hd))
            self.dWh += h_prev.T @ dz
            self.db += dz.sum(axis=0, keepdims=True)

            dh_next = dz @ self.Wh.T
            dX[:, t, :] = dz @ self.Wx.T

        self.dWx /= N
        self.dWh /= N
        self.db /= N

        return dX

    def params(self):
        return {"Wx": self.Wx, "Wh": self.Wh, "b": self.b}

    def grads(self):
        return {"Wx": self.dWx, "Wh": self.dWh, "b": self.db}

    def __repr__(self):
        return (
            f"SimpleRNN({self.input_dim}->{self.hidden_dim}, "
            f"return_sequences={self.return_sequences})"
        )


# ============================================================
# LSTM
# ============================================================
class LSTM(Layer):
    """
    Long Short-Term Memory (Hochreiter & Schmidhuber, 1997).

    Portões:
        i_t = σ(x_t Wxi + h_{t-1} Whi + bi)     (input)
        f_t = σ(x_t Wxf + h_{t-1} Whf + bf)     (forget)
        o_t = σ(x_t Wxo + h_{t-1} Who + bo)     (output)
        g_t = tanh(x_t Wxg + h_{t-1} Whg + bg)  (candidate)

    Estado:
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t         (cell)
        h_t = o_t ⊙ tanh(c_t)                    (hidden)

    Os quatro conjuntos de pesos são concatenados em Wx, Wh e b para
    eficiência. A bias da forget gate é inicializada em 1.0 (prática
    comum para preservar o gradiente no início do treino).
    """

    def __init__(self, input_dim, hidden_dim, return_sequences=False,
                 init="xavier", seed=None):
        rng = np.random.default_rng(seed)
        if init == "he":
            scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        elif init == "xavier":
            scale = np.sqrt(1.0 / (input_dim + hidden_dim))
        else:
            scale = 0.01

        self.Wx = (rng.standard_normal((input_dim, 4 * hidden_dim)) * scale).astype(np.float64)
        self.Wh = (rng.standard_normal((hidden_dim, 4 * hidden_dim)) * scale).astype(np.float64)
        self.b = np.zeros((1, 4 * hidden_dim), dtype=np.float64)

        # forget gate bias = 1
        self.b[0, hidden_dim:2 * hidden_dim] = 1.0

        self.dWx = None
        self.dWh = None
        self.db = None

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        self._cache = None

    def forward(self, X, training=True):
        assert X.ndim == 3, f"LSTM espera (N, T, D), recebeu shape {X.shape}"
        N, T, D = X.shape
        assert D == self.input_dim

        Hd = self.hidden_dim

        h = np.zeros((N, Hd))
        c = np.zeros((N, Hd))

        H_all = np.zeros((N, T, Hd))
        C_all = np.zeros((N, T, Hd))
        i_all = np.zeros((N, T, Hd))
        f_all = np.zeros((N, T, Hd))
        o_all = np.zeros((N, T, Hd))
        g_all = np.zeros((N, T, Hd))

        for t in range(T):
            z = X[:, t, :] @ self.Wx + h @ self.Wh + self.b

            i = _sigmoid(z[:, :Hd])
            f = _sigmoid(z[:, Hd:2 * Hd])
            o = _sigmoid(z[:, 2 * Hd:3 * Hd])
            g = np.tanh(z[:, 3 * Hd:4 * Hd])

            c = f * c + i * g
            h = o * np.tanh(c)

            H_all[:, t, :] = h
            C_all[:, t, :] = c
            i_all[:, t, :] = i
            f_all[:, t, :] = f
            o_all[:, t, :] = o
            g_all[:, t, :] = g

        self._cache = (X, H_all, C_all, i_all, f_all, o_all, g_all)
        return H_all if self.return_sequences else H_all[:, -1, :]

    def backward(self, dout):
        X, H_all, C_all, i_all, f_all, o_all, g_all = self._cache
        N, T, D = X.shape
        Hd = self.hidden_dim

        if self.return_sequences:
            dH = dout
        else:
            dH = np.zeros((N, T, Hd))
            dH[:, -1, :] = dout

        dX = np.zeros_like(X)
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)

        dh_next = np.zeros((N, Hd))
        dc_next = np.zeros((N, Hd))

        for t in range(T - 1, -1, -1):
            dh = dH[:, t, :] + dh_next

            o = o_all[:, t, :]
            i = i_all[:, t, :]
            f = f_all[:, t, :]
            g = g_all[:, t, :]
            c = C_all[:, t, :]
            c_prev = C_all[:, t - 1, :] if t > 0 else np.zeros((N, Hd))
            h_prev = H_all[:, t - 1, :] if t > 0 else np.zeros((N, Hd))

            tanh_c = np.tanh(c)

            # h = o ⊙ tanh(c)
            do = dh * tanh_c
            dc = dh * o * (1 - tanh_c ** 2) + dc_next

            # c = f ⊙ c_prev + i ⊙ g
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev = dc * f

            # Através das ativações
            di_pre = di * i * (1 - i)
            df_pre = df * f * (1 - f)
            do_pre = do * o * (1 - o)
            dg_pre = dg * (1 - g ** 2)

            dz = np.concatenate([di_pre, df_pre, do_pre, dg_pre], axis=1)

            self.dWx += X[:, t, :].T @ dz
            self.dWh += h_prev.T @ dz
            self.db += dz.sum(axis=0, keepdims=True)

            dh_next = dz @ self.Wh.T
            dc_next = dc_prev

            dX[:, t, :] = dz @ self.Wx.T

        self.dWx /= N
        self.dWh /= N
        self.db /= N

        return dX

    def params(self):
        return {"Wx": self.Wx, "Wh": self.Wh, "b": self.b}

    def grads(self):
        return {"Wx": self.dWx, "Wh": self.dWh, "b": self.db}

    def __repr__(self):
        return (
            f"LSTM({self.input_dim}->{self.hidden_dim}, "
            f"return_sequences={self.return_sequences})"
        )
