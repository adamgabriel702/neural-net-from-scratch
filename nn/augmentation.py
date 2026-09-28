"""
Transformações de data augmentation para imagens (N, C, H, W).

Cada transform é uma classe com `__call__(X)` que recebe e devolve
um array com o mesmo shape. Todas operam em batch (N, C, H, W) e são
puramente NumPy — sem PIL, sem torchvision.

Uso típico:
    aug = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomShift(max_shift=2),
    ])
    X_aug = aug(X_batch)
"""

import numpy as np


class Transform:
    """Interface base. Toda transform recebe e devolve (N, C, H, W)."""

    def __call__(self, X):
        raise NotImplementedError


class Compose(Transform):
    """Aplica uma sequência de transforms na ordem."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, X):
        for t in self.transforms:
            X = t(X)
        return X

    def __repr__(self):
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"


class RandomHorizontalFlip(Transform):
    """Espelha horizontalmente com probabilidade `p`, por amostra."""

    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, X):
        N = X.shape[0]
        mask = self.rng.random(N) < self.p
        if not mask.any():
            return X
        X = X.copy()
        X[mask] = X[mask][:, :, :, ::-1]
        return X

    def __repr__(self):
        return f"RandomHorizontalFlip(p={self.p})"


class RandomShift(Transform):
    """
    Desloca a imagem por ±max_shift pixels em H e W, por amostra,
    preenchendo as bordas com zeros.
    """

    def __init__(self, max_shift=1, seed=None):
        self.max_shift = max_shift
        self.rng = np.random.default_rng(seed)

    def __call__(self, X):
        N, C, H, W = X.shape
        out = np.zeros_like(X)
        dh = self.rng.integers(-self.max_shift, self.max_shift + 1, size=N)
        dw = self.rng.integers(-self.max_shift, self.max_shift + 1, size=N)

        for i in range(N):
            h_src_start = max(0, -dh[i])
            h_src_end = H - max(0, dh[i])
            w_src_start = max(0, -dw[i])
            w_src_end = W - max(0, dw[i])

            h_dst_start = max(0, dh[i])
            h_dst_end = H - max(0, -dh[i])
            w_dst_start = max(0, dw[i])
            w_dst_end = W - max(0, -dw[i])

            out[i, :, h_dst_start:h_dst_end, w_dst_start:w_dst_end] = X[
                i, :, h_src_start:h_src_end, w_src_start:w_src_end
            ]
        return out

    def __repr__(self):
        return f"RandomShift(max_shift={self.max_shift})"


class RandomRotation90(Transform):
    """
    Rotaciona 0°, 90°, 180° ou 270° com probabilidade `p`, por amostra.
    Útil em datasets onde orientação não importa (mas não no MNIST,
    porque 6 e 9 viram um ao outro).
    """

    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, X):
        N = X.shape[0]
        mask = self.rng.random(N) < self.p
        k = self.rng.integers(1, 4, size=N)  # 1, 2 ou 3 rotações de 90°
        X = X.copy()
        for i in range(N):
            if mask[i]:
                X[i] = np.rot90(X[i], k=k[i], axes=(1, 2))
        return X

    def __repr__(self):
        return f"RandomRotation90(p={self.p})"


class GaussianNoise(Transform):
    """Adiciona ruído gaussiano `N(0, sigma²)` aos pixels."""

    def __init__(self, sigma=0.05, seed=None):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def __call__(self, X):
        return X + self.rng.normal(0.0, self.sigma, size=X.shape)

    def __repr__(self):
        return f"GaussianNoise(sigma={self.sigma})"
