import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


@pytest.fixture
def xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, y


@pytest.fixture
def multiclass_data():
    """3 classes em blobs gaussianos separáveis — usado nos testes de convergência."""
    from nn.datasets import make_blobs
    return make_blobs(n_samples=300, n_features=5, n_classes=3, seed=0)


@pytest.fixture
def random_onehot():
    """(X, y_oh) aleatórios para testes de gradiente."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 4))
    y_idx = rng.integers(0, 3, size=6)
    y_oh = np.zeros((6, 3))
    y_oh[np.arange(6), y_idx] = 1.0
    return X, y_oh