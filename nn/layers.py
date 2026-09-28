"""
Camadas da rede neural implementadas do zero com NumPy.

Convenções:
- Todo layer implementa `forward(X, training=True)` e `backward(dout)`.
- `backward(dout)` recebe o gradiente da loss em relação à SAÍDA do layer
  e retorna o gradiente em relação à ENTRADA do layer.
- Layers com parâmetros treináveis expõem `params()` e `grads()`.
- Layers com estado não-treinável (ex.: running_mean em BatchNorm) expõem
  `buffers()`, que é salvo/carregado junto com o modelo.
- Operações que agem "sobre as classes" (softmax, etc.) usam `axis=-1`,
  funcionando uniformemente para (N, C), (N, T, C) e (N, H, W, C).
"""

import numpy as np

from .activations import ACTIVATIONS


# ============================================================
# Classe base
# ============================================================
class Layer:
    """Interface base para todas as camadas."""

    def forward(self, X, training=True):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def params(self):
        return {}

    def grads(self):
        return {}

    def buffers(self):
        return {}


# ============================================================
# Camada totalmente conectada
# ============================================================
class Dense(Layer):
    """
    Camada densa: y = X @ W + b

    Parâmetros
    ----------
    in_dim : int
    out_dim : int
    init : {"xavier", "he", "small"}
    seed : int ou None
    """

    def __init__(self, in_dim, out_dim, init="xavier", seed=None):
        rng = np.random.default_rng(seed)

        if init == "he":
            scale = np.sqrt(2.0 / in_dim)
        elif init == "xavier":
            scale = np.sqrt(1.0 / in_dim)
        elif init == "small":
            scale = 0.01
        else:
            scale = np.sqrt(1.0 / in_dim)

        self.W = (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float64)
        self.b = np.zeros((1, out_dim), dtype=np.float64)

        self.dW = None
        self.db = None
        self._X = None

    def forward(self, X, training=True):
        self._X = X
        return X @ self.W + self.b

    def backward(self, dout):
        m = self._X.shape[0]
        self.dW = (self._X.T @ dout) / m
        self.db = np.sum(dout, axis=0, keepdims=True) / m
        return dout @ self.W.T

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}


# ============================================================
# Camada de ativação
# ============================================================
class ActivationLayer(Layer):
    """
    Aplica uma função de ativação não-linear.

    Suporta: sigmoid, relu, tanh, softmax, linear.
    Para softmax, o backward usa o Jacobiano completo em `axis=-1`.
    Em `Sequential` com loss CCE, o cálculo é fundido (A - y).
    """

    def __init__(self, name):
        if name not in ACTIVATIONS:
            raise ValueError(
                f"Ativação '{name}' desconhecida. "
                f"Opções: {list(ACTIVATIONS.keys())}"
            )
        self.name = name
        self.fn, self.fn_back = ACTIVATIONS[name]
        self._Z = None
        self._A = None

    def forward(self, Z, training=True):
        self._Z = Z
        self._A = self.fn(Z)
        return self._A

    def backward(self, dout):
        if self.name == "softmax":
            # Jacobiano: dL/dz_i = a_i * (dL/da_i - sum_j dL/da_j * a_j)
            # `axis=-1` para suportar tensores 3D/4D.
            dot = np.sum(dout * self._A, axis=-1, keepdims=True)
            return self._A * (dout - dot)
        return dout * self.fn_back(self._Z, self._A)

    def __repr__(self):
        return f"ActivationLayer({self.name})"


# ============================================================
# Dropout (invertido)
# ============================================================
class Dropout(Layer):
    """
    Dropout invertido (Srivastava et al., 2014).

    Durante o treino, zera fração `p` das ativações e escala o resto por
    1/(1-p), de modo que a expectativa permaneça igual. No inference,
    não faz nada.
    """

    def __init__(self, p=0.5, seed=None):
        if not 0.0 <= p < 1.0:
            raise ValueError("p deve estar em [0, 1).")
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.mask = None

    def forward(self, X, training=True):
        if not training or self.p == 0.0:
            self.mask = None
            return X
        keep_prob = 1.0 - self.p
        self.mask = (self.rng.random(X.shape) < keep_prob) / keep_prob
        return X * self.mask

    def backward(self, dout):
        if self.mask is None:
            return dout
        return dout * self.mask

    def __repr__(self):
        return f"Dropout(p={self.p})"


# ============================================================
# Batch Normalization (para MLP: normaliza por feature)
# ============================================================
class BatchNorm(Layer):
    """
    Batch Normalization (Ioffe & Szegedy, 2015).

    Normaliza cada feature pelas estatísticas do mini-batch durante o
    treino e usa médias móveis no inference. Parâmetros treináveis
    (gamma, beta) e buffers não-treináveis (running_mean, running_var).
    """

    def __init__(self, dim, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((1, dim), dtype=np.float64)
        self.beta = np.zeros((1, dim), dtype=np.float64)
        self.dgamma = None
        self.dbeta = None

        self.running_mean = np.zeros((1, dim), dtype=np.float64)
        self.running_var = np.ones((1, dim), dtype=np.float64)

        self.momentum = momentum
        self.eps = eps
        self._cache = None

    def forward(self, X, training=True):
        if training:
            mu = X.mean(axis=0, keepdims=True)
            var = X.var(axis=0, keepdims=True)
            std_inv = 1.0 / np.sqrt(var + self.eps)
            x_hat = (X - mu) * std_inv

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self._cache = (X, x_hat, mu, var, std_inv)
        else:
            std_inv = 1.0 / np.sqrt(self.running_var + self.eps)
            x_hat = (X - self.running_mean) * std_inv

        return self.gamma * x_hat + self.beta

    def backward(self, dout):
        X, x_hat, mu, var, std_inv = self._cache
        m = X.shape[0]

        self.dgamma = np.sum(dout * x_hat, axis=0, keepdims=True) / m
        self.dbeta = np.sum(dout, axis=0, keepdims=True) / m

        dx_hat = dout * self.gamma

        dvar = np.sum(dx_hat * (X - mu) * -0.5 * (std_inv ** 3), axis=0, keepdims=True)
        dmu = np.sum(dx_hat * -std_inv, axis=0, keepdims=True) + \
              dvar * np.mean(-2.0 * (X - mu), axis=0, keepdims=True)

        dX = dx_hat * std_inv + dvar * 2.0 * (X - mu) / m + dmu / m
        return dX

    def params(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def grads(self):
        return {"gamma": self.dgamma, "beta": self.dbeta}

    def buffers(self):
        return {
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }

    def __repr__(self):
        return f"BatchNorm(dim={self.gamma.shape[1]})"


# ============================================================
# BatchNorm2D (channels-first: N, C, H, W)
# ============================================================
class BatchNorm2D(Layer):
    """
    Batch Normalization para Conv2D.

    Normaliza cada canal separadamente pelas estatísticas do mini-batch
    sobre os eixos (N, H, W) durante o treino, e usa médias móveis no
    inference. Parâmetros treináveis (gamma, beta) por canal.
    """

    def __init__(self, num_channels, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((1, num_channels, 1, 1), dtype=np.float64)
        self.beta = np.zeros((1, num_channels, 1, 1), dtype=np.float64)
        self.dgamma = None
        self.dbeta = None

        self.running_mean = np.zeros((1, num_channels, 1, 1), dtype=np.float64)
        self.running_var = np.ones((1, num_channels, 1, 1), dtype=np.float64)

        self.momentum = momentum
        self.eps = eps
        self._cache = None

    def forward(self, X, training=True):
        if training:
            mu = X.mean(axis=(0, 2, 3), keepdims=True)
            var = X.var(axis=(0, 2, 3), keepdims=True)
            std_inv = 1.0 / np.sqrt(var + self.eps)
            x_hat = (X - mu) * std_inv

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self._cache = (X, x_hat, mu, var, std_inv)
        else:
            std_inv = 1.0 / np.sqrt(self.running_var + self.eps)
            x_hat = (X - self.running_mean) * std_inv

        return self.gamma * x_hat + self.beta

    def backward(self, dout):
        X, x_hat, mu, var, std_inv = self._cache
        N = X.shape[0]
        m_per_channel = X.shape[0] * X.shape[2] * X.shape[3]

        # gamma / beta: mean-reduced (divisão por N)
        self.dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True) / N
        self.dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True) / N

        dx_hat = dout * self.gamma

        dvar = np.sum(
            dx_hat * (X - mu) * -0.5 * (std_inv ** 3),
            axis=(0, 2, 3), keepdims=True,
        )
        dmu = np.sum(dx_hat * -std_inv, axis=(0, 2, 3), keepdims=True) + \
              dvar * np.mean(-2.0 * (X - mu), axis=(0, 2, 3), keepdims=True)

        dX = dx_hat * std_inv + dvar * 2.0 * (X - mu) / m_per_channel + dmu / m_per_channel
        return dX

    def params(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def grads(self):
        return {"gamma": self.dgamma, "beta": self.dbeta}

    def buffers(self):
        return {
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }

    def __repr__(self):
        return f"BatchNorm2D(C={self.gamma.shape[1]})"


# ============================================================
# Conv2D (channels-first: N, C, H, W)
# ============================================================
class Conv2D(Layer):
    """
    Convolução 2D com im2col + matmul.

    Entrada:  X (N, C_in, H, W)
    Peso:     W (C_out, C_in, kH, kW)
    Viés:     b (C_out, 1)
    Saída:    (N, C_out, H_out, W_out)

    H_out = (H + 2p - kH) // stride + 1
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        init="he",
        seed=None,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH, self.kW = kernel_size
        self.sH, self.sW = stride
        self.padding = padding

        rng = np.random.default_rng(seed)
        fan_in = in_channels * self.kH * self.kW
        if init == "he":
            scale = np.sqrt(2.0 / fan_in)
        elif init == "xavier":
            scale = np.sqrt(1.0 / fan_in)
        else:
            scale = 0.01

        self.W = (
            rng.standard_normal((out_channels, in_channels, self.kH, self.kW)) * scale
        ).astype(np.float64)
        self.b = np.zeros((out_channels, 1), dtype=np.float64)

        self.dW = None
        self.db = None
        self._cache = None

    def _im2col(self, X_pad, H_out, W_out):
        N, C, _, _ = X_pad.shape
        cols = np.zeros((N, C, self.kH, self.kW, H_out, W_out), dtype=X_pad.dtype)
        for i in range(self.kH):
            i_end = i + self.sH * H_out
            for j in range(self.kW):
                j_end = j + self.sW * W_out
                cols[:, :, i, j, :, :] = X_pad[
                    :, :, i:i_end:self.sH, j:j_end:self.sW
                ]
        return cols.reshape(N, C * self.kH * self.kW, H_out * W_out)

    def _col2im(self, dcols, X_shape, H_out, W_out):
        N, C, H, W = X_shape
        H_pad = H + 2 * self.padding
        W_pad = W + 2 * self.padding
        dX_pad = np.zeros((N, C, H_pad, W_pad), dtype=dcols.dtype)

        dcols_r = dcols.reshape(N, C, self.kH, self.kW, H_out, W_out)
        for i in range(self.kH):
            i_end = i + self.sH * H_out
            for j in range(self.kW):
                j_end = j + self.sW * W_out
                dX_pad[:, :, i:i_end:self.sH, j:j_end:self.sW] += dcols_r[:, :, i, j, :, :]

        if self.padding > 0:
            return dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dX_pad

    def forward(self, X, training=True):
        N, C, H, W = X.shape
        assert C == self.in_channels, (
            f"Conv2D esperava {self.in_channels} canais, recebeu {C}"
        )

        H_out = (H + 2 * self.padding - self.kH) // self.sH + 1
        W_out = (W + 2 * self.padding - self.kW) // self.sW + 1

        if self.padding > 0:
            X_pad = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
            )
        else:
            X_pad = X

        cols = self._im2col(X_pad, H_out, W_out)
        W_col = self.W.reshape(self.out_channels, -1)

        out = W_col @ cols
        out = out + self.b.reshape(1, self.out_channels, 1)
        out = out.reshape(N, self.out_channels, H_out, W_out)

        self._cache = (X.shape, cols, H_out, W_out)
        return out

    def backward(self, dout):
        X_shape, cols, H_out, W_out = self._cache
        N, C, H, W = X_shape
        P = H_out * W_out

        dout_flat = dout.reshape(N, self.out_channels, P)

        self.db = dout_flat.sum(axis=(0, 2), keepdims=False).reshape(self.out_channels, 1) / N
        self.dW = np.einsum("nop,njp->oj", dout_flat, cols) / N
        self.dW = self.dW.reshape(self.W.shape)

        W_col = self.W.reshape(self.out_channels, -1)
        dcols = W_col.T @ dout_flat
        dX = self._col2im(dcols, X_shape, H_out, W_out)
        return dX

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}

    def __repr__(self):
        return (
            f"Conv2D({self.in_channels}->{self.out_channels}, "
            f"k={self.kH}x{self.kW}, s={self.sH}, p={self.padding})"
        )


# ============================================================
# MaxPool2D
# ============================================================
class MaxPool2D(Layer):
    """Max pooling 2D não sobreposto (ou com stride customizado)."""

    def __init__(self, pool_size=2, stride=None):
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        self.pH, self.pW = pool_size
        if stride is None:
            stride = pool_size
        if isinstance(stride, int):
            stride = (stride, stride)
        self.sH, self.sW = stride
        self._cache = None

    def forward(self, X, training=True):
        N, C, H, W = X.shape
        H_out = (H - self.pH) // self.sH + 1
        W_out = (W - self.pW) // self.sW + 1

        out = np.zeros((N, C, H_out, W_out), dtype=X.dtype)
        argmax = np.zeros((N, C, H_out, W_out), dtype=np.int64)

        for i in range(H_out):
            hs = i * self.sH
            for j in range(W_out):
                ws = j * self.sW
                window = X[:, :, hs:hs + self.pH, ws:ws + self.pW]
                window_flat = window.reshape(N, C, -1)
                argmax[:, :, i, j] = np.argmax(window_flat, axis=-1)
                out[:, :, i, j] = np.max(window_flat, axis=-1)

        self._cache = (X.shape, argmax)
        return out

    def backward(self, dout):
        X_shape, argmax = self._cache
        N, C, H, W = X_shape
        H_out, W_out = dout.shape[2], dout.shape[3]

        dX = np.zeros(X_shape, dtype=dout.dtype)

        n_idx = np.arange(N)[:, None]
        c_idx = np.arange(C)[None, :]

        for i in range(H_out):
            hs = i * self.sH
            for j in range(W_out):
                ws = j * self.sW
                di = argmax[:, :, i, j] // self.pW
                dj = argmax[:, :, i, j] % self.pW
                dX[n_idx, c_idx, hs + di, ws + dj] += dout[:, :, i, j]

        return dX

    def __repr__(self):
        return f"MaxPool2D(pool={self.pH}x{self.pW}, stride={self.sH})"


# ============================================================
# Flatten
# ============================================================
class Flatten(Layer):
    """(N, C, H, W) → (N, C*H*W). Inverso no backward."""

    def __init__(self):
        self._input_shape = None

    def forward(self, X, training=True):
        self._input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._input_shape)

    def __repr__(self):
        return "Flatten()"