import numpy as np
from nn.losses import Loss


def test_mse_zero():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert Loss.mse(y, y) == 0


def test_bce_symmetry():
    y = np.array([[1.0], [0.0]])
    a = Loss.binary_cross_entropy(y, np.array([[0.9], [0.1]]))
    b = Loss.binary_cross_entropy(y, np.array([[0.1], [0.9]]))
    assert a < b


def test_cce_one_hot():
    y = np.array([[1.0, 0.0, 0.0]])
    a = Loss.categorical_cross_entropy(y, np.array([[0.9, 0.05, 0.05]]))
    b = Loss.categorical_cross_entropy(y, np.array([[0.3, 0.4, 0.3]]))
    assert a < b
