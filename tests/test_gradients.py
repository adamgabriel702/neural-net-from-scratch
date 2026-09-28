"""
Verificação de gradiente numérico para MLP.

Cada combinação (ativação, loss) é testada contra a derivada numérica.
Este arquivo é a "prova" de que o backprop está correto.
"""
import numpy as np
import pytest

from nn import Sequential, Dense, ActivationLayer, Adam
from tests.helpers import check_gradients


def _build_mlp(activation, loss, in_dim, hid, out_dim, seed=0):
    model = Sequential()
    model.add(Dense(in_dim, hid, init="xavier"))
    model.add(ActivationLayer(activation))
    model.add(Dense(hid, out_dim, init="xavier"))
    if loss == "bce":
        model.add(ActivationLayer("sigmoid"))
    elif loss == "cce":
        model.add(ActivationLayer("softmax"))
    else:
        model.add(ActivationLayer("linear"))
    model.compile(loss=loss, optimizer=Adam(lr=0.01))
    return model


def _make_data(n, in_dim, out_dim, one_hot=False, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, in_dim))
    if one_hot:
        idx = rng.integers(0, out_dim, size=n)
        y = np.zeros((n, out_dim))
        y[np.arange(n), idx] = 1.0
    else:
        y = (rng.random((n, out_dim)) > 0.5).astype(float)
    return X, y


def _run(activation, loss, out_dim, one_hot):
    X, y = _make_data(6, 4, out_dim, one_hot=one_hot)
    model = _build_mlp(activation, loss, in_dim=4, hid=5, out_dim=out_dim)
    check_gradients(model, X, y, layer_indices=[0, 2])


def test_gradients_sigmoid_bce():
    _run("sigmoid", "bce", 1, one_hot=False)


def test_gradients_tanh_bce():
    _run("tanh", "bce", 1, one_hot=False)


def test_gradients_relu_mse():
    _run("relu", "mse", 2, one_hot=False)


def test_gradients_tanh_mse():
    _run("tanh", "mse", 2, one_hot=False)


def test_gradients_softmax_cce():
    _run("relu", "cce", 3, one_hot=True)