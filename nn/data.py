"""
DataLoader: batching com shuffle e augmentation opcional.

Uso típico:
    loader = DataLoader(X, y_oh, batch_size=32, shuffle=True, transform=aug)
    for X_batch, y_batch in loader:
        ...

Ou integrado ao `Sequential.fit`:
    model.fit(loader, epochs=15, ...)
"""

import numpy as np


class DataLoader:
    """
    Itera mini-batches `(X_batch, y_batch)`.

    Parâmetros
    ----------
    X, y : arrays
        Mesma primeira dimensão (número de amostras).
    batch_size : int
    shuffle : bool
        Reembaralha antes de cada `__iter__`.
    transform : callable ou None
        Aplicado em `X_batch` a cada batch (só em treino). Espera e
        devolve (N, ...).
    seed : int ou None
        Semente para reprodutibilidade.
    drop_last : bool
        Se True, descarta o último batch se for incompleto.
    """

    def __init__(self, X, y, batch_size=32, shuffle=True,
                 transform=None, seed=None, drop_last=False):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        assert self.X.shape[0] == self.y.shape[0], (
            "X e y devem ter o mesmo número de amostras"
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.rng = np.random.default_rng(seed)
        self.drop_last = drop_last

    def __len__(self):
        n = self.X.shape[0]
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.X.shape[0]
        idx = self.rng.permutation(n) if self.shuffle else np.arange(n)

        for s in range(0, n, self.batch_size):
            batch = idx[s:s + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue

            Xb = self.X[batch]
            yb = self.y[batch]

            if self.transform is not None:
                Xb = self.transform(Xb)

            yield Xb, yb

    def __repr__(self):
        return (
            f"DataLoader(n={len(self.X)}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, transform={self.transform})"
        )
