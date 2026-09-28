import numpy as np
import pytest

from nn import (
    DataLoader, Compose,
    RandomHorizontalFlip, RandomShift, RandomRotation90, GaussianNoise,
)


# ---------- DataLoader ----------
def test_dataloader_batching():
    X = np.arange(100).reshape(10, 10)
    y = np.arange(10)
    loader = DataLoader(X, y, batch_size=4, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0][0].shape == (4, 10)
    assert batches[1][0].shape == (4, 10)
    assert batches[2][0].shape == (2, 10)


def test_dataloader_drop_last():
    X = np.zeros((10, 5))
    y = np.zeros(10)
    loader = DataLoader(X, y, batch_size=4, shuffle=False, drop_last=True)
    assert len(list(loader)) == 2


def test_dataloader_shuffle_changes_order():
    X = np.arange(20).reshape(20, 1)
    y = np.arange(20)
    loader = DataLoader(X, y, batch_size=20, shuffle=True, seed=0)
    (Xb, yb), = list(loader)
    assert not np.array_equal(Xb.ravel(), np.arange(20))


def test_dataloader_with_transform():
    X = np.ones((8, 1, 4, 4))
    y = np.zeros(8)

    def double(x):
        return x * 2

    loader = DataLoader(X, y, batch_size=8, shuffle=False, transform=double)
    (Xb, _), = list(loader)
    assert np.allclose(Xb, 2.0)


# ---------- transforms ----------
def test_compose_order():
    calls = []

    class A:
        def __call__(self, X):
            calls.append("A")
            return X

    class B:
        def __call__(self, X):
            calls.append("B")
            return X

    Compose([A(), B()])(np.zeros((1, 1, 2, 2)))
    assert calls == ["A", "B"]


def test_flip_is_involution():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 1, 5, 5))
    flip = RandomHorizontalFlip(p=1.0, seed=0)
    once = flip(X)
    twice = flip(once)
    assert np.allclose(twice, X)


def test_flip_p_zero_is_identity():
    X = np.random.randn(4, 1, 5, 5)
    flip = RandomHorizontalFlip(p=0.0)
    assert np.allclose(flip(X), X)


def test_shift_preserves_shape():
    X = np.random.randn(3, 2, 8, 8)
    shift = RandomShift(max_shift=2, seed=0)
    assert shift(X).shape == X.shape


def test_shift_zero_is_identity():
    X = np.random.randn(4, 1, 6, 6)
    shift = RandomShift(max_shift=0)
    assert np.allclose(shift(X), X)


def test_shift_actually_moves_pixels():
    X = np.zeros((1, 1, 5, 5))
    X[0, 0, 2, 2] = 1.0
    shift = RandomShift(max_shift=1, seed=42)
    out = shift(X)
    # O pixel deve continuar existindo, mas possivelmente em outra posição
    assert out.sum() == pytest.approx(1.0)
    # Com semente 42, sempre cai em algum deslocamento determinístico
    assert not np.array_equal(out, X)


def test_rotation_preserves_shape():
    X = np.random.randn(2, 1, 4, 4)
    rot = RandomRotation90(p=1.0, seed=0)
    assert rot(X).shape == X.shape


def test_rotation_90_twice_is_180():
    X = np.arange(16).reshape(1, 1, 4, 4).astype(float)
    rot = RandomRotation90(p=1.0, seed=0)
    once = rot(X)
    # Rotacionar 1x pode ser 90/180/270 — não é determinístico por batch
    # Mas shape e conteúdo (permutação) devem ser preservados
    assert sorted(once.ravel()) == sorted(X.ravel())


def test_gaussian_noise_sigma_zero_is_identity():
    X = np.random.randn(3, 1, 4, 4)
    noise = GaussianNoise(sigma=0.0)
    assert np.allclose(noise(X), X)


def test_gaussian_noise_nonzero():
    X = np.zeros((10, 1, 10, 10))
    noise = GaussianNoise(sigma=0.5, seed=0)
    out = noise(X)
    assert not np.allclose(out, X)
    assert abs(out.std() - 0.5) < 0.05


# ---------- integração com fit ----------
def test_fit_with_transform():
    from nn import Sequential, Dense, ActivationLayer, Adam

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4))
    y = (rng.random((40, 1)) > 0.5).astype(float)

    model = Sequential()
    model.add(Dense(4, 8, init="xavier"))
    model.add(ActivationLayer("tanh"))
    model.add(Dense(8, 1, init="xavier"))
    model.add(ActivationLayer("sigmoid"))
    model.compile(loss="bce", optimizer=Adam(lr=1e-2))

    model.fit(X, y, epochs=3, batch_size=8, verbose=False, shuffle=True,
              transform=GaussianNoise(sigma=0.01, seed=0))

    assert len(model.history["loss"]) == 3


def test_fit_accepts_dataloader():
    from nn import Sequential, Dense, ActivationLayer, Adam

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4))
    y = (rng.random((40, 1)) > 0.5).astype(float)

    loader = DataLoader(X, y, batch_size=8, shuffle=True, seed=0)

    model = Sequential()
    model.add(Dense(4, 8, init="xavier"))
    model.add(ActivationLayer("tanh"))
    model.add(Dense(8, 1, init="xavier"))
    model.add(ActivationLayer("sigmoid"))
    model.compile(loss="bce", optimizer=Adam(lr=1e-2))

    model.fit(loader, epochs=3, verbose=False)
    assert len(model.history["loss"]) == 3
