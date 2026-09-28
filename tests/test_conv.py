import numpy as np
import pytest

from nn import Sequential, Conv2D, MaxPool2D, Flatten, Dense, ActivationLayer, Adam
from tests.helpers import check_gradients


# ---------- shapes ----------
def test_conv2d_output_shape():
    conv = Conv2D(3, 8, kernel_size=3, stride=1, padding=0)
    X = np.random.randn(4, 3, 10, 10)
    assert conv.forward(X).shape == (4, 8, 8, 8)


def test_conv2d_padding_same():
    conv = Conv2D(3, 8, kernel_size=3, stride=1, padding=1)
    X = np.random.randn(4, 3, 10, 10)
    assert conv.forward(X).shape == (4, 8, 10, 10)


def test_conv2d_stride():
    conv = Conv2D(3, 8, kernel_size=3, stride=2, padding=1)
    X = np.random.randn(4, 3, 10, 10)
    assert conv.forward(X).shape == (4, 8, 5, 5)


def test_conv2d_backward_shape():
    conv = Conv2D(3, 8, kernel_size=3, stride=1, padding=1)
    X = np.random.randn(4, 3, 10, 10)
    conv.forward(X)
    dout = np.random.randn(4, 8, 10, 10)
    dX = conv.backward(dout)
    assert dX.shape == X.shape
    assert conv.dW.shape == conv.W.shape
    assert conv.db.shape == conv.b.shape


def test_conv2d_bias_broadcast():
    """Regression: bias (C_out, 1) precisa broadcastar contra (N, C_out, P)."""
    conv = Conv2D(1, 4, kernel_size=3, stride=1, padding=1)
    conv.b = np.arange(4, dtype=float).reshape(4, 1)
    X = np.zeros((2, 1, 6, 6))
    out = conv.forward(X)
    assert out.shape == (2, 4, 6, 6)
    for c in range(4):
        assert np.allclose(out[:, c, :, :], conv.b[c, 0])


def test_conv2d_identity_kernel():
    """Kernel 1×1 identidade devolve a própria entrada."""
    conv = Conv2D(1, 1, kernel_size=1)
    conv.W = np.array([[[[1.0]]]])
    conv.b = np.zeros((1, 1))
    X = np.random.randn(3, 1, 5, 5)
    assert np.allclose(conv.forward(X), X)


def test_maxpool2d_shape():
    pool = MaxPool2D(2)
    X = np.random.randn(4, 3, 8, 8)
    assert pool.forward(X).shape == (4, 3, 4, 4)


def test_maxpool2d_values():
    pool = MaxPool2D(2)
    X = np.array([[[[1, 2], [3, 4]]]], dtype=float)
    out = pool.forward(X)
    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == 4


def test_maxpool2d_backward_routing():
    """Gradiente vai só para o elemento que foi máximo."""
    pool = MaxPool2D(2)
    X = np.array([[[[1, 2], [3, 4]]]], dtype=float)
    pool.forward(X)
    dout = np.ones((1, 1, 1, 1))
    dX = pool.backward(dout)
    expected = np.array([[[[0, 0], [0, 1]]]], dtype=float)
    assert np.allclose(dX, expected)


def test_flatten_roundtrip():
    from nn.layers import Flatten as FlattenLayer
    flat = FlattenLayer()
    X = np.random.randn(2, 3, 4, 5)
    out = flat.forward(X)
    assert out.shape == (2, 60)
    assert np.allclose(flat.backward(out), X)


# ---------- gradiente numérico ----------
def _conv_out_hw(H, kernel, stride, padding):
    return (H + 2 * padding - kernel) // stride + 1


def _build_conv_model(padding, stride, in_channels=1, out_channels=2,
                      H=6, kernel=3, seed=0):
    rng = np.random.default_rng(seed)
    N = 2
    C = in_channels

    X = rng.standard_normal((N, C, H, H))

    # One-hot garante que cada linha soma 1 (válido para softmax + CCE).
    # Binário aleatório podia gerar linhas [0, 0, 0] → NaN no gradiente.
    y_idx = rng.integers(0, 3, size=N)
    y = np.zeros((N, 3))
    y[np.arange(N), y_idx] = 1.0

    H_conv = _conv_out_hw(H, kernel, stride, padding)
    H_pool = H_conv // 2
    dense_in = out_channels * H_pool * H_pool

    model = Sequential()
    model.add(Conv2D(C, out_channels, kernel_size=kernel,
                     stride=stride, padding=padding))
    model.add(ActivationLayer("relu"))
    model.add(MaxPool2D(2))
    model.add(Flatten())
    model.add(Dense(dense_in, 3, init="xavier"))
    model.add(ActivationLayer("softmax"))

    model.compile(loss="cce", optimizer=Adam(lr=1e-2))
    return model, X, y


def _check_conv(padding, stride, in_channels=1, out_channels=2):
    model, X, y = _build_conv_model(padding, stride, in_channels, out_channels)
    check_gradients(model, X, y, layer_indices=[0], keys=["W", "b"])


def test_conv2d_gradients_pad0_stride1():
    _check_conv(padding=0, stride=1)


def test_conv2d_gradients_pad1_stride1():
    _check_conv(padding=1, stride=1)


def test_conv2d_gradients_pad1_stride2():
    _check_conv(padding=1, stride=2)


def test_conv2d_gradients_multichannel():
    """3 canais de entrada, 4 de saída — pega bugs de layout."""
    _check_conv(padding=1, stride=1, in_channels=3, out_channels=4)