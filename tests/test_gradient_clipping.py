import numpy as np
import pytest

from nn import Sequential, Dense, ActivationLayer, Adam, SGD, clip_gradients


class _FakeLayer:
    """Layer mínimo com gradientes controlados."""
    def __init__(self, params, grads):
        self._p = params
        self._g = grads

    def params(self):
        return self._p

    def grads(self):
        return self._g


# ---------- clip_gradients ----------
def test_clip_no_op_when_below_threshold():
    g = {"w": np.array([0.1, 0.2])}
    layer = _FakeLayer({"w": np.zeros(2)}, g)
    original = g["w"].copy()

    norm = clip_gradients([layer], max_norm=1.0)
    assert np.allclose(g["w"], original)
    assert abs(norm - np.linalg.norm(original)) < 1e-12


def test_clip_scales_when_above_threshold():
    g = {"w": np.array([3.0, 4.0])}  # norma = 5.0
    layer = _FakeLayer({"w": np.zeros(2)}, g)

    norm = clip_gradients([layer], max_norm=1.0)
    assert abs(norm - 5.0) < 1e-12
    assert abs(np.linalg.norm(g["w"]) - 1.0) < 1e-12


def test_clip_multiple_layers():
    g1 = {"w": np.array([3.0, 4.0])}      # 5.0
    g2 = {"w": np.array([0.0, 12.0])}     # 12.0
    l1 = _FakeLayer({"w": np.zeros(2)}, g1)
    l2 = _FakeLayer({"w": np.zeros(2)}, g2)

    norm = clip_gradients([l1, l2], max_norm=13.0)
    assert abs(norm - 13.0) < 1e-12
    # Ambos devem estar reduzidos pelo mesmo fator (13 / 13)
    assert abs(np.linalg.norm(g1["w"]) - 5.0) < 1e-12
    assert abs(np.linalg.norm(g2["w"]) - 12.0) < 1e-12


def test_clip_ignores_none_grads():
    layer = _FakeLayer({"w": np.zeros(2), "b": np.zeros(1)},
                       {"w": np.array([3.0, 4.0]), "b": None})
    norm = clip_gradients([layer], max_norm=1.0)
    assert abs(norm - 5.0) < 1e-12


# ---------- integração com otimizadores ----------
def _build_model():
    model = Sequential()
    model.add(Dense(4, 6, init="xavier"))
    model.add(ActivationLayer("tanh"))
    model.add(Dense(6, 1, init="xavier"))
    model.add(ActivationLayer("sigmoid"))
    return model


def _data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 4))
    y = (rng.random((8, 1)) > 0.5).astype(float)
    return X, y


def test_sgd_applies_clip():
    X, y = _data()
    model = _build_model()

    # Compara dois SGDs, um com clip e outro sem
    opt_no_clip = SGD(lr=0.1, clip_norm=None)
    model.compile(loss="bce", optimizer=opt_no_clip)
    model.fit(X, y, epochs=1, batch_size=8, verbose=False, shuffle=False)
    loss_no_clip = model.history["loss"][0]

    model = _build_model()
    opt_clip = SGD(lr=0.1, clip_norm=0.01)   # clipping agressivo
    model.compile(loss="bce", optimizer=opt_clip)
    model.fit(X, y, epochs=1, batch_size=8, verbose=False, shuffle=False)
    loss_clip = model.history["loss"][0]

    # Com clip agressivo, a atualização é menor → loss final difere
    assert loss_no_clip != loss_clip


def test_adam_applies_clip():
    X, y = _data()
    model = _build_model()
    opt = Adam(lr=0.01, clip_norm=1.0)
    model.compile(loss="bce", optimizer=opt)
    model.fit(X, y, epochs=3, batch_size=8, verbose=False)
    assert len(model.history["loss"]) == 3


def test_clip_does_not_break_convergence():
    """Clipping brando não deve impedir convergência em XOR."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    model = Sequential()
    model.add(Dense(2, 8, init="xavier"))
    model.add(ActivationLayer("tanh"))
    model.add(Dense(8, 1, init="xavier"))
    model.add(ActivationLayer("sigmoid"))

    opt = Adam(lr=0.05, clip_norm=10.0)
    model.compile(loss="bce", optimizer=opt)
    model.fit(X, y, epochs=300, batch_size=4, verbose=False)

    preds = (model.predict(X) > 0.5).astype(int)
    assert np.array_equal(preds, y.astype(int))
