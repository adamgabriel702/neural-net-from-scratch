"""
Datasets para exemplos e experimentos.

Todos os datasets retornam `(X_train, y_train, X_test, y_test)` em
numpy.float64 / int64. Sem dependência de scikit-learn.
"""

import os
import gzip
import urllib.request
import numpy as np


# ============================================================
# MNIST
# ============================================================
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def _download(url, path):
    if not os.path.exists(path):
        print(f"Baixando {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)


def _load_mnist_images(path):
    with gzip.open(path, "rb") as f:
        assert int.from_bytes(f.read(4), "big") == 2051, "magic number inesperado"
        n = int.from_bytes(f.read(4), "big")
        r = int.from_bytes(f.read(4), "big")
        c = int.from_bytes(f.read(4), "big")
        buf = f.read(n * r * c)
        return (
            np.frombuffer(buf, dtype=np.uint8)
            .reshape(n, 1, r, c)
            .astype(np.float64)
        )


def _load_mnist_labels(path):
    with gzip.open(path, "rb") as f:
        assert int.from_bytes(f.read(4), "big") == 2049, "magic number inesperado"
        n = int.from_bytes(f.read(4), "big")
        return np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int64)


def load_mnist(data_dir="data/mnist", flatten=False, normalize=True):
    """
    Carrega o MNIST.

    Parâmetros
    ----------
    data_dir : str
        Onde os arquivos .gz são baixados/cacheados.
    flatten : bool
        Se True, X tem shape (N, 784). Se False, (N, 1, 28, 28) — pronto
        para `Conv2D`.
    normalize : bool
        Se True, escala para [0, 1] em float64.

    Retorna
    -------
    (X_train, y_train, X_test, y_test)
        X: float64; y: int64.
    """
    os.makedirs(data_dir, exist_ok=True)
    paths = {k: os.path.join(data_dir, fn) for k, fn in MNIST_FILES.items()}
    for k, fn in MNIST_FILES.items():
        _download(MNIST_URL + fn, paths[k])

    Xtr = _load_mnist_images(paths["train_x"])
    ytr = _load_mnist_labels(paths["train_y"])
    Xte = _load_mnist_images(paths["test_x"])
    yte = _load_mnist_labels(paths["test_y"])

    if flatten:
        Xtr = Xtr.reshape(Xtr.shape[0], -1)
        Xte = Xte.reshape(Xte.shape[0], -1)

    if normalize:
        Xtr = Xtr / 255.0
        Xte = Xte / 255.0

    return Xtr, ytr, Xte, yte


# ============================================================
# Datasets sintéticos (úteis para testes/CI)
# ============================================================
def make_blobs(n_samples=300, n_features=5, n_classes=3, seed=0, center=3.0):
    """
    Blobs gaussianos separáveis por eixo. Uma classe por eixo.

    Retorna `(X, y)` — y é int64, não one-hot.
    """
    rng = np.random.default_rng(seed)
    per_class = n_samples // n_classes

    centers = np.zeros((n_classes, n_features))
    for c in range(n_classes):
        centers[c, c % n_features] = center

    X_parts, y_parts = [], []
    for c in range(n_classes):
        X_parts.append(rng.standard_normal((per_class, n_features)) + centers[c])
        y_parts.append(np.full(per_class, c, dtype=np.int64))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_xor(seed=0):
    """Retorna o dataset XOR clássico: X (4, 2), y (4, 1)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)
    return X, y
