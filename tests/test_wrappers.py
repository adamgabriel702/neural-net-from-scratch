import numpy as np
import pytest

from nn import (
    Sequential, TimeDistributed, SimpleRNN, LSTM,
    Dense, ActivationLayer, Adam,
)
from tests.helpers import check_gradients


# ---------- shapes ----------
def test_timedistributed_dense_shape():
    td = TimeDistributed(Dense(5, 3))
    X = np.random.randn(4, 6, 5)
    out = td.forward(X)
    assert out.shape == (4, 6, 3)


def test_timedistributed_backward_shape():
    td = TimeDistributed(Dense(5, 3))
    X = np.random.randn(4, 6, 5)
    td.forward(X)
    dX = td.backward(np.random.randn(4, 6, 3))
    assert dX.shape == X.shape


def test_timedistributed_delegates_params():
    dense = Dense(4, 2)
    td = TimeDistributed(dense)
    # Mesmos OBJETOS numpy (não cópias)
    assert td.params()["W"] is dense.params()["W"]
    assert td.params()["b"] is dense.params()["b"]
    assert td.grads()["W"] is dense.grads()["W"]
    assert td.grads()["b"] is dense.grads()["b"]


def test_timedistributed_rejects_non_layer():
    with pytest.raises(TypeError):
        TimeDistributed("not a layer")


def test_timedistributed_rejects_2d():
    td = TimeDistributed(Dense(4, 2))
    with pytest.raises(ValueError):
        td.forward(np.random.randn(8, 4))


def test_timedistributed_matches_manual_loop():
    """O resultado deve ser idêntico a aplicar a Dense timestep por timestep."""
    dense = Dense(5, 3, seed=0)
    td = TimeDistributed(dense)

    X = np.random.randn(2, 4, 5)
    out_td = td.forward(X)

    manual = np.stack(
        [dense.forward(X[:, t, :]) for t in range(X.shape[1])],
        axis=1,
    )
    assert np.allclose(out_td, manual)


# ---------- gradiente numérico ----------
def test_timedistributed_dense_gradients():
    X = np.random.randn(3, 5, 4)
    y_idx = np.random.default_rng(0).integers(0, 3, size=3)
    y = np.zeros((3, 3))
    y[np.arange(3), y_idx] = 1.0

    model = Sequential()
    model.add(TimeDistributed(Dense(4, 6, init="xavier", seed=0)))
    model.add(ActivationLayer("tanh"))
    model.add(TimeDistributed(Dense(6, 3, init="xavier", seed=1)))
    model.add(ActivationLayer("softmax"))
    model.compile(loss="cce", optimizer=Adam(lr=1e-2))

    # A saída é (N, T, 3). y_td replica o one-hot sobre T.
    y_td = np.tile(y[:, None, :], (1, 5, 1))

    model.forward(X)
    model.backward(y_td)

    for key in ("W", "b"):
        analytic = model.layers[0].grads()[key]
        numeric = _numerical_grad(model, X, y_td, layer_idx=0, key=key,
                                  eps=1e-6, training=False)
        denom = np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12
        diff = np.linalg.norm(analytic - numeric) / denom
        assert diff < 1e-5, (
            f"TimeDistributed(Dense) grad errado (key={key}), "
            f"erro relativo = {diff:.2e}"
        )


def test_timedistributed_gradients_second_layer():
    X = np.random.randn(3, 4, 5)
    y_idx = np.random.default_rng(1).integers(0, 3, size=(3, 4))
    y = np.zeros((3, 4, 3))
    for i in range(3):
        for t in range(4):
            y[i, t, y_idx[i, t]] = 1.0

    model = Sequential()
    model.add(TimeDistributed(Dense(5, 6, init="xavier", seed=0)))
    model.add(ActivationLayer("tanh"))
    model.add(TimeDistributed(Dense(6, 3, init="xavier", seed=1)))
    model.add(ActivationLayer("softmax"))
    model.compile(loss="cce", optimizer=Adam(lr=1e-2))

    model.forward(X)
    model.backward(y)

    for key in ("W", "b"):
        analytic = model.layers[2].grads()[key]
        numeric = _numerical_grad(model, X, y, layer_idx=2, key=key,
                                  eps=1e-6, training=False)
        denom = np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12
        diff = np.linalg.norm(analytic - numeric) / denom
        assert diff < 1e-5, (
            f"TimeDistributed(Dense) 2ª camada grad errado (key={key}), "
            f"erro relativo = {diff:.2e}"
        )


def _numerical_grad(model, X, y, layer_idx, key, eps=1e-6, training=False):
    p = model.layers[layer_idx].params()[key]
    grad = np.zeros_like(p)
    it = np.nditer(p, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = p[idx]

        p[idx] = orig + eps
        loss_plus = model.loss_fn(y, model.forward(X, training=training))

        p[idx] = orig - eps
        loss_minus = model.loss_fn(y, model.forward(X, training=training))

        p[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


# ---------- integração com RNN ----------
def test_rnn_with_timedistributed_output():
    """
    RNN(return_seq=True) → TimeDistributed(Dense) → softmax.
    Cada timestep produz uma predição própria.
    """
    rng = np.random.default_rng(0)
    N, T, D, C = 16, 5, 4, 3

    X = rng.standard_normal((N, T, D))
    y_idx = rng.integers(0, C, size=(N, T))
    y = np.zeros((N, T, C))
    for i in range(N):
        for t in range(T):
            y[i, t, y_idx[i, t]] = 1.0

    model = Sequential()
    model.add(SimpleRNN(D, 8, return_sequences=True, seed=0))
    model.add(TimeDistributed(Dense(8, C, init="xavier", seed=1)))
    model.add(ActivationLayer("softmax"))
    model.compile(loss="cce", optimizer=Adam(lr=0.01))

    model.fit(X, y, epochs=400, batch_size=4, verbose=False)

    preds = model.predict(X)
    assert preds.shape == (N, T, C)

    pred_classes = np.argmax(preds, axis=-1)
    acc = (pred_classes == y_idx).mean()
    assert acc > 0.5