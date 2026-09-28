import numpy as np
from nn.layers import Dense, Dropout


def test_dense_shapes():
    layer = Dense(4, 3)
    X = np.random.randn(10, 4)
    out = layer.forward(X)
    assert out.shape == (10, 3)


def test_dense_backward_shape():
    layer = Dense(4, 3)
    X = np.random.randn(10, 4)
    layer.forward(X)
    dZ = np.random.randn(10, 3)
    dX = layer.backward(dZ)
    assert dX.shape == (10, 4)
    assert layer.dW.shape == (4, 3)
    assert layer.db.shape == (1, 3)


def test_dropout_train_vs_eval():
    drop = Dropout(p=0.5)
    X = np.ones((100, 100))
    out_train = drop.forward(X, training=True)
    out_eval = drop.forward(X, training=False)
    assert not np.allclose(out_train, X)
    assert np.allclose(out_eval, X)


def test_dropout_scaling():
    drop = Dropout(p=0.5)
    X = np.ones((1000, 1000))
    out = drop.forward(X, training=True)
    # Escala inversa: média deve ficar ~1
    assert abs(out.mean() - 1.0) < 0.1
