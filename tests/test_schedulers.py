import numpy as np
import pytest

from nn import Adam, StepLR, ExponentialLR, CosineAnnealing, WarmupCosine
from nn.callbacks import EarlyStopping, ModelCheckpoint, History


class _FakeModel:
    """Modelo mínimo para testar callbacks sem treinar."""
    def __init__(self):
        self.optimizer = Adam(lr=1e-3)
        self.history = {"loss": []}
        self._saved = []

    def save(self, path):
        self._saved.append(path)


# ---------- schedulers ----------
def test_step_lr():
    opt = Adam(lr=1.0)
    sched = StepLR(opt, step_size=2, gamma=0.5)
    assert sched.get_lr() == 1.0
    sched.step()
    assert sched.get_lr() == 1.0
    sched.step()
    assert sched.get_lr() == 0.5
    sched.step()
    assert sched.get_lr() == 0.5
    sched.step()
    assert sched.get_lr() == 0.25


def test_exponential_lr():
    opt = Adam(lr=1.0)
    sched = ExponentialLR(opt, gamma=0.9)
    for _ in range(5):
        sched.step()
    assert np.isclose(sched.get_lr(), 0.9 ** 5)


def test_cosine_annealing_endpoints():
    opt = Adam(lr=1.0)
    sched = CosineAnnealing(opt, t_max=10, eta_min=0.0)
    # t=0 → lr = base_lr
    assert np.isclose(sched.get_lr(), 1.0)
    # t=T → lr = eta_min
    for _ in range(10):
        sched.step()
    assert np.isclose(sched.get_lr(), 0.0, atol=1e-9)
    # t>T → clampado em eta_min
    sched.step()
    assert np.isclose(sched.get_lr(), 0.0, atol=1e-9)


def test_warmup_cosine():
    opt = Adam(lr=1.0)
    sched = WarmupCosine(opt, warmup_epochs=3, t_max=10, eta_min=0.1)
    # warmup: t=0 → lr/3; t=1 → 2lr/3; t=2 → lr
    assert np.isclose(sched.get_lr(), 1/3)
    sched.step()
    assert np.isclose(sched.get_lr(), 2/3)
    sched.step()
    assert np.isclose(sched.get_lr(), 1.0)


def test_scheduler_updates_optimizer():
    opt = Adam(lr=1.0)
    sched = StepLR(opt, step_size=1, gamma=0.5)
    sched.step()
    assert opt.lr == 0.5
    sched.step()
    assert opt.lr == 0.25


# ---------- callbacks ----------
def test_early_stopping_min():
    model = _FakeModel()
    es = EarlyStopping(monitor="val_loss", patience=2, mode="min")

    es.on_epoch_end(model, 1, {"val_loss": 0.5})   # melhora (best)
    assert not es.should_stop
    es.on_epoch_end(model, 2, {"val_loss": 0.4})   # melhora
    assert not es.should_stop
    es.on_epoch_end(model, 3, {"val_loss": 0.45})  # piora (wait=1)
    es.on_epoch_end(model, 4, {"val_loss": 0.46})  # piora (wait=2)
    es.on_epoch_end(model, 5, {"val_loss": 0.47})  # wait=3 → para? patience=2 → sim
    assert es.should_stop


def test_early_stopping_max():
    model = _FakeModel()
    es = EarlyStopping(monitor="accuracy", patience=1, mode="max")

    es.on_epoch_end(model, 1, {"accuracy": 0.8})
    assert not es.should_stop
    es.on_epoch_end(model, 2, {"accuracy": 0.75})  # piora
    es.on_epoch_end(model, 3, {"accuracy": 0.74})  # piora → para
    assert es.should_stop


def test_model_checkpoint_saves_on_improvement():
    model = _FakeModel()
    ckpt = ModelCheckpoint("/tmp/x.npz", monitor="val_loss", mode="min", verbose=False)

    ckpt.on_epoch_end(model, 1, {"val_loss": 0.5})
    assert model._saved == ["/tmp/x.npz"]
    ckpt.on_epoch_end(model, 2, {"val_loss": 0.6})  # pior
    assert model._saved == ["/tmp/x.npz"]           # não salvou de novo
    ckpt.on_epoch_end(model, 3, {"val_loss": 0.4})  # melhor
    assert model._saved == ["/tmp/x.npz", "/tmp/x.npz"]


def test_history_callback():
    model = _FakeModel()
    hist = History()
    hist.on_epoch_end(model, 1, {"loss": 0.5, "val_loss": 0.6})
    hist.on_epoch_end(model, 2, {"loss": 0.4, "val_loss": 0.5})
    assert hist.history["loss"] == [0.5, 0.4]
    assert hist.history["val_loss"] == [0.6, 0.5]


# ---------- integração com fit ----------
def test_fit_integrates_scheduler_and_callbacks():
    from nn import Sequential, Dense, ActivationLayer

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4))
    y = (rng.random((40, 1)) > 0.5).astype(float)

    model = Sequential()
    model.add(Dense(4, 8, init="xavier"))
    model.add(ActivationLayer("tanh"))
    model.add(Dense(8, 1, init="xavier"))
    model.add(ActivationLayer("sigmoid"))

    opt = Adam(lr=1e-2)
    model.compile(loss="bce", optimizer=opt)

    sched = StepLR(opt, step_size=2, gamma=0.5)
    hist = History()

    model.fit(
        X, y,
        epochs=5, batch_size=8,
        verbose=False, shuffle=False,
        scheduler=sched,
        callbacks=[hist],
    )

    assert len(hist.history["loss"]) == 5
    assert len(hist.history["lr"]) == 5
    # step_size=2, gamma=0.5 → lrs: 1e-2, 1e-2, 5e-3, 5e-3, 2.5e-3
    assert np.isclose(hist.history["lr"][0], 1e-2)
    assert np.isclose(hist.history["lr"][2], 5e-3)
