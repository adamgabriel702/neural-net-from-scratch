import numpy as np
from nn.activations import Activation


def test_sigmoid_range():
    z = np.linspace(-20, 20, 100)
    s = Activation.sigmoid(z)
    assert np.all(s >= 0) and np.all(s <= 1)


def test_sigmoid_at_zero():
    s = Activation.sigmoid(np.array([0.0]))
    assert np.isclose(s[0], 0.5)


def test_sigmoid_no_overflow():
    z = np.array([-1000.0, 1000.0])
    s = Activation.sigmoid(z)
    assert np.all(np.isfinite(s))


def test_relu():
    z = np.array([-3, -1, 0, 1, 3], dtype=float)
    assert np.allclose(Activation.relu(z), [0, 0, 0, 1, 3])


def test_softmax_sums_to_one():
    z = np.random.randn(10, 5)
    s = Activation.softmax(z)
    assert np.allclose(s.sum(axis=1), 1.0)


def test_softmax_stability():
    z = np.array([[1000.0, 1000.0, 1000.0]])
    s = Activation.softmax(z)
    assert np.all(np.isfinite(s))
    assert np.allclose(s, 1 / 3)


def test_tanh_deriv():
    z = np.linspace(-3, 3, 50)
    a = Activation.tanh(z)
    d = Activation.tanh_backward(z, a)
    eps = 1e-6
    d_num = (Activation.tanh(z + eps) - Activation.tanh(z - eps)) / (2 * eps)
    assert np.allclose(d, d_num, atol=1e-6)