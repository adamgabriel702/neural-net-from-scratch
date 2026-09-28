import numpy as np
from nn import Sequential, Dense, ActivationLayer, Adam, SGD
from nn import metrics


def _xor_model():
    m = Sequential()
    m.add(Dense(2, 8, init="xavier"))
    m.add(ActivationLayer("tanh"))
    m.add(Dense(8, 1, init="xavier"))
    m.add(ActivationLayer("sigmoid"))
    return m


def test_xor_converges(xor_data):
    X, y = xor_data
    model = _xor_model()
    model.compile(loss="bce", optimizer=Adam(lr=0.05))
    model.fit(X, y, epochs=300, batch_size=4, verbose=False)
    preds = (model.predict(X) > 0.5).astype(int)
    assert np.array_equal(preds, y.astype(int))


def test_multiclass_converges(multiclass_data):
    from nn.utils import one_hot, train_test_split, standardize
    X, y = multiclass_data
    X, _, _ = standardize(X)
    y_oh = one_hot(y, 3)
    Xtr, Xte, ytr, yte = train_test_split(X, y_oh, test_size=0.2)

    model = Sequential()
    model.add(Dense(5, 16, init="he"))
    model.add(ActivationLayer("relu"))
    model.add(Dense(16, 3, init="xavier"))
    model.add(ActivationLayer("softmax"))
    model.compile(loss="cce", optimizer=Adam(lr=0.01))
    model.fit(Xtr, ytr, epochs=200, batch_size=16, verbose=False)

    acc = metrics.accuracy(yte, model.predict(Xte))
    assert acc > 0.85


def test_save_load(tmp_path, xor_data):
    X, y = xor_data
    model = _xor_model()
    model.compile(loss="bce", optimizer=Adam(lr=0.05))
    model.fit(X, y, epochs=100, batch_size=4, verbose=False)
    p1 = model.predict(X)

    path = tmp_path / "model.npz"
    model.save(str(path))

    model2 = _xor_model()
    model2.load(str(path))
    p2 = model2.predict(X)

    assert np.allclose(p1, p2, atol=1e-10)
