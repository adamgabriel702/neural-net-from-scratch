import numpy as np
import pytest

from nn import (
    Sequential, Conv2D, BatchNorm, BatchNorm2D, Dense, ActivationLayer,
    Adam, Flatten,
)
from tests.helpers import check_gradients


# ---------- formas e stats ----------
def test_batchnorm2d_shape():
    bn = BatchNorm2D(4)
    X = np.random.randn(8, 4, 6, 6)
    assert bn.forward(X).shape == X.shape


def test_batchnorm2d_running_stats_update():
    bn = BatchNorm2D(3)
    X = np.random.randn(4, 3, 5, 5) * 2 + 1.5
    bn.forward(X, training=True)
    empirical_mean = X.mean(axis=(0, 2, 3), keepdims=True)
    assert not np.allclose(bn.running_mean, 0.0)
    assert np.allclose(bn.running_mean, 0.1 * empirical_mean, atol=1e-6)


def test_batchnorm2d_inference_uses_running_stats():
    bn = BatchNorm2D(3, momentum=0.0)
    X = np.random.randn(8, 3, 4, 4) * 3 + 2
    bn.forward(X, training=True)
    out_eval = bn.forward(X, training=False)
    assert out_eval.shape == X.shape


# ---------- gradientes ----------
def _make_bn2d_model(shape, seed=0):
    rng = np.random.default_rng(seed)
    N, C, H, W = shape

    X = rng.standard_normal(shape)
    y = (rng.random((N, 3)) > 0.5).astype(float)
    y = y / y.sum(axis=1, keepdims=True)

    dense_in = C * H * W

    model = Sequential()
    model.add(Conv2D(C, C, kernel_size=1))
    model.add(BatchNorm2D(C))
    model.add(ActivationLayer("relu"))
    model.add(Flatten())
    model.add(Dense(dense_in, 3, init="xavier"))
    model.add(ActivationLayer("softmax"))
    model.compile(loss="cce", optimizer=Adam(lr=1e-2))
    return model, X, y


def test_batchnorm2d_gradients():
    model, X, y = _make_bn2d_model((2, 3, 4, 4))
    check_gradients(model, X, y, layer_indices=[1], keys=["gamma", "beta"],
                    training=True)


def test_batchnorm2d_gradients_multichannel():
    model, X, y = _make_bn2d_model((3, 5, 3, 3), seed=1)
    check_gradients(model, X, y, layer_indices=[1], keys=["gamma", "beta"],
                    training=True)


def test_batchnorm1d_gradients():
    rng = np.random.default_rng(0)
    N, F = 6, 4
    X = rng.standard_normal((N, F))
    y = (rng.random((N, 3)) > 0.5).astype(float)
    y = y / y.sum(axis=1, keepdims=True)

    model = Sequential()
    model.add(Dense(F, F, init="xavier"))
    model.add(BatchNorm(F))
    model.add(ActivationLayer("relu"))
    model.add(Dense(F, 3, init="xavier"))
    model.add(ActivationLayer("softmax"))
    model.compile(loss="cce", optimizer=Adam(lr=1e-2))

    check_gradients(model, X, y, layer_indices=[1], keys=["gamma", "beta"],
                    training=True)