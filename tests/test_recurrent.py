import numpy as np
import pytest

from nn import (
    Sequential, SimpleRNN, LSTM, Dense, ActivationLayer, Adam,
)
from tests.helpers import check_gradients


# ---------- shapes ----------
def test_simple_rnn_last_output_shape():
    rnn = SimpleRNN(input_dim=4, hidden_dim=6, return_sequences=False)
    X = np.random.randn(3, 5, 4)
    assert rnn.forward(X).shape == (3, 6)


def test_simple_rnn_sequences_output_shape():
    rnn = SimpleRNN(input_dim=4, hidden_dim=6, return_sequences=True)
    X = np.random.randn(3, 5, 4)
    assert rnn.forward(X).shape == (3, 5, 6)


def test_simple_rnn_backward_shapes():
    rnn = SimpleRNN(input_dim=4, hidden_dim=6)
    X = np.random.randn(3, 5, 4)
    rnn.forward(X)
    dX = rnn.backward(np.random.randn(3, 6))
    assert dX.shape == X.shape
    assert rnn.dWx.shape == rnn.Wx.shape
    assert rnn.dWh.shape == rnn.Wh.shape
    assert rnn.db.shape == rnn.b.shape


def test_simple_rnn_zero_initial_state():
    """Com h_0 = 0, a primeira saída não depende do resto da sequência."""
    rnn = SimpleRNN(input_dim=3, hidden_dim=5, return_sequences=True, seed=0)
    X = np.random.randn(1, 4, 3)
    H = rnn.forward(X)

    # Muda apenas o último timestep. A saída do t=0 deve permanecer idêntica.
    X2 = X.copy()
    X2[0, -1, :] = np.random.randn(3)
    H2 = rnn.forward(X2)
    assert np.allclose(H[0, 0, :], H2[0, 0, :])


def test_lstm_last_output_shape():
    lstm = LSTM(input_dim=4, hidden_dim=6, return_sequences=False)
    X = np.random.randn(3, 5, 4)
    assert lstm.forward(X).shape == (3, 6)


def test_lstm_sequences_output_shape():
    lstm = LSTM(input_dim=4, hidden_dim=6, return_sequences=True)
    X = np.random.randn(3, 5, 4)
    assert lstm.forward(X).shape == (3, 5, 6)


def test_lstm_backward_shapes():
    lstm = LSTM(input_dim=4, hidden_dim=6)
    X = np.random.randn(3, 5, 4)
    lstm.forward(X)
    dX = lstm.backward(np.random.randn(3, 6))
    assert dX.shape == X.shape
    assert lstm.dWx.shape == lstm.Wx.shape
    assert lstm.dWh.shape == lstm.Wh.shape
    assert lstm.db.shape == lstm.b.shape


def test_lstm_forget_bias_init():
    """Bias da forget gate deve começar em 1.0."""
    lstm = LSTM(input_dim=4, hidden_dim=6)
    Hd = 6
    assert np.allclose(lstm.b[0, Hd:2 * Hd], 1.0)
    assert np.allclose(lstm.b[0, :Hd], 0.0)
    assert np.allclose(lstm.b[0, 2 * Hd:3 * Hd], 0.0)
    assert np.allclose(lstm.b[0, 3 * Hd:], 0.0)


# ---------- gradiente numérico (BPTT) ----------
def _make_seq_data(n=4, T=5, D=4, C=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, D))
    y_idx = rng.integers(0, C, size=n)
    y = np.zeros((n, C))
    y[np.arange(n), y_idx] = 1.0
    return X, y


def _make_rnn_classifier(rnn_cls, **kwargs):
    model = Sequential()
    model.add(rnn_cls(**kwargs))
    model.add(Dense(kwargs["hidden_dim"], 3, init="xavier"))
    model.add(ActivationLayer("softmax"))
    model.compile(loss="cce", optimizer=Adam(lr=1e-2))
    return model


def test_simple_rnn_gradients():
    X, y = _make_seq_data()
    model = _make_rnn_classifier(
        SimpleRNN, input_dim=4, hidden_dim=5, return_sequences=False, seed=0,
    )
    check_gradients(model, X, y, layer_indices=[0], keys=["Wx", "Wh", "b"])


def test_simple_rnn_gradients_longer_sequence():
    X, y = _make_seq_data(n=3, T=8, D=4, seed=1)
    model = _make_rnn_classifier(
        SimpleRNN, input_dim=4, hidden_dim=4, return_sequences=False, seed=1,
    )
    check_gradients(model, X, y, layer_indices=[0], keys=["Wx", "Wh", "b"])


def test_lstm_gradients():
    X, y = _make_seq_data()
    model = _make_rnn_classifier(
        LSTM, input_dim=4, hidden_dim=5, return_sequences=False, seed=0,
    )
    check_gradients(model, X, y, layer_indices=[0], keys=["Wx", "Wh", "b"])


def test_lstm_gradients_longer_sequence():
    X, y = _make_seq_data(n=3, T=8, D=4, seed=1)
    model = _make_rnn_classifier(
        LSTM, input_dim=4, hidden_dim=4, return_sequences=False, seed=1,
    )
    check_gradients(model, X, y, layer_indices=[0], keys=["Wx", "Wh", "b"])


# ---------- integração ----------
def test_rnn_can_remember_first_step():
    """
    A rede tem que classificar a sequência baseada no PRIMEIRO timestep.
    Isso força o RNN a propagar a informação do início até o fim.
    """
    rng = np.random.default_rng(0)
    N, T, D = 80, 5, 3
    X = rng.standard_normal((N, T, D))
    y = (X[:, 0, 0] > 0).astype(int)
    y_oh = np.zeros((N, 2))
    y_oh[np.arange(N), y] = 1.0

    model = Sequential()
    model.add(SimpleRNN(D, 12, return_sequences=False, seed=0))
    model.add(ActivationLayer("tanh"))
    model.add(Dense(12, 2, init="xavier"))
    model.add(ActivationLayer("softmax"))
    model.compile(loss="cce", optimizer=Adam(lr=0.01))
    model.fit(X, y_oh, epochs=400, batch_size=16, verbose=False)

    preds = model.predict(X)
    acc = (np.argmax(preds, axis=1) == y).mean()
    assert acc > 0.85


def test_lstm_can_remember_first_step():
    rng = np.random.default_rng(0)
    N, T, D = 80, 5, 3
    X = rng.standard_normal((N, T, D))
    y = (X[:, 0, 0] > 0).astype(int)
    y_oh = np.zeros((N, 2))
    y_oh[np.arange(N), y] = 1.0

    model = Sequential()
    model.add(LSTM(D, 12, return_sequences=False, seed=0))
    model.add(ActivationLayer("tanh"))
    model.add(Dense(12, 2, init="xavier"))
    model.add(ActivationLayer("softmax"))
    model.compile(loss="cce", optimizer=Adam(lr=0.01))
    model.fit(X, y_oh, epochs=400, batch_size=16, verbose=False)

    preds = model.predict(X)
    acc = (np.argmax(preds, axis=1) == y).mean()
    assert acc > 0.85
