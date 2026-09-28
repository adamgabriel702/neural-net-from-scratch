"""
Modelo `Sequential` — API inspirada no Keras.
"""

import os
import numpy as np

from .layers import Dense, ActivationLayer, BatchNorm, Dropout
from .losses import LOSSES
from .callbacks import Callback


class Sequential:
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.loss_back = None
        self.loss_name = None
        self.optimizer = None
        self._fused_softmax_cce = False
        self.history = {"loss": []}

    # ---------------- construção ----------------
    def add(self, layer):
        self.layers.append(layer)
        return self

    def compile(self, loss="bce", optimizer=None):
        if loss not in LOSSES:
            raise ValueError(
                f"Loss '{loss}' desconhecida. Opções: {list(LOSSES.keys())}"
            )
        self.loss_fn, self.loss_back = LOSSES[loss]
        self.loss_name = loss
        self.optimizer = optimizer

        self._fused_softmax_cce = (
            loss == "cce"
            and len(self.layers) >= 1
            and isinstance(self.layers[-1], ActivationLayer)
            and self.layers[-1].name == "softmax"
        )
        return self

    # ---------------- forward / backward ----------------
    def forward(self, X, training=True):
        A = X
        for layer in self.layers:
            A = layer.forward(A, training=training)
        return A

    def backward(self, y):
        A_last = self.layers[-1]._A

        if self._fused_softmax_cce:
            # softmax + CCE fundidos: dL/dZ = (A - y), sum-reduced
            dout = A_last - y
            layers_to_backprop = self.layers[:-1]
        else:
            dout = self.loss_back(y, A_last)
            layers_to_backprop = self.layers

        for layer in reversed(layers_to_backprop):
            dout = layer.backward(dout)

    # ---------------- treino ----------------
    def fit(
        self,
        X,
        y,
        epochs=100,
        batch_size=32,
        verbose=True,
        shuffle=True,
        validation_data=None,
        scheduler=None,
        callbacks=None,
    ):
        """
        Treina o modelo.

        Parâmetros extras
        -----------------
        scheduler : _LRScheduler ou None
            Se fornecido, tem seu `step()` chamado ao final de cada época.
        callbacks : list de Callback ou None
            Executados ao final de cada época com `(model, epoch, logs)`.
        """
        if self.optimizer is None:
            raise RuntimeError(
                "Chame `compile(loss=..., optimizer=...)` antes de `fit`."
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        callbacks = callbacks or []

        n = X.shape[0]
        log_every = max(1, epochs // 10)

        for cb in callbacks:
            cb.on_train_begin(self)

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n) if shuffle else np.arange(n)
            Xs, ys = X[idx], y[idx]

            for s in range(0, n, batch_size):
                Xb = Xs[s:s + batch_size]
                yb = ys[s:s + batch_size]

                self.forward(Xb, training=True)
                self.backward(yb)
                self.optimizer.step(self.layers)

            # Loss no dataset completo em modo eval
            y_pred = self.forward(X, training=False)
            loss = float(self.loss_fn(y, y_pred))
            self.history["loss"].append(loss)

            logs = {"loss": loss}

            if validation_data is not None:
                Xv, yv = validation_data
                Xv = np.asarray(Xv, dtype=np.float64)
                yv = np.asarray(yv, dtype=np.float64)
                val_pred = self.forward(Xv, training=False)
                val_loss = float(self.loss_fn(yv, val_pred))
                self.history.setdefault("val_loss", []).append(val_loss)
                logs["val_loss"] = val_loss

            if scheduler is not None:
                scheduler.step()
                logs["lr"] = self.optimizer.lr

            if verbose and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
                msg = f"Epoch {epoch:4d}/{epochs} | loss: {loss:.6f}"
                if "val_loss" in logs:
                    msg += f" | val_loss: {logs['val_loss']:.6f}"
                if "lr" in logs:
                    msg += f" | lr: {logs['lr']:.2e}"
                print(msg)

            # Callbacks
            for cb in callbacks:
                cb.on_epoch_end(self, epoch, logs)

            # Early stopping pode interromper
            if any(getattr(cb, "should_stop", False) for cb in callbacks):
                if verbose:
                    stopped = [cb for cb in callbacks if getattr(cb, "should_stop", False)]
                    for cb in stopped:
                        print(f"  [early stop] {cb.__class__.__name__} na época {epoch}")
                break

        for cb in callbacks:
            cb.on_train_end(self)

        return self.history

    # ---------------- predição ----------------
    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.forward(X, training=False)

    def predict_classes(self, X):
        preds = self.predict(X)
        if preds.ndim > 1 and preds.shape[1] > 1:
            return np.argmax(preds, axis=1)
        return (preds.ravel() > 0.5).astype(int)

    def evaluate(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        return float(self.loss_fn(y, self.predict(X)))

    # ---------------- persistência ----------------
    def save(self, path):
        state = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.params().items():
                if v is not None:
                    state[f"layer{i}_param_{k}"] = v
            for k, v in layer.buffers().items():
                state[f"layer{i}_buffer_{k}"] = v
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **state)

    def load(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            for k in layer.params():
                key = f"layer{i}_param_{k}"
                if key in data:
                    layer.params()[k][...] = data[key]
            for k in layer.buffers():
                key = f"layer{i}_buffer_{k}"
                if key in data:
                    layer.buffers()[k][...] = data[key]
        return self

    # ---------------- utilitários ----------------
    def summary(self):
        print("=" * 70)
        print(f"{'Layer':<35}{'Output Shape':<20}{'Params':>12}")
        print("=" * 70)

        total = 0
        for layer in self.layers:
            n_params = sum(v.size for v in layer.params().values() if v is not None)
            total += n_params
            name = layer.__class__.__name__
            if isinstance(layer, ActivationLayer):
                name = f"Activation({layer.name})"
            elif isinstance(layer, Dense):
                name = f"Dense({layer.W.shape[0]}->{layer.W.shape[1]})"

            shape = getattr(layer, "_A", None)
            shape_str = str(shape.shape) if shape is not None else "-"
            print(f"{name:<35}{shape_str:<20}{n_params:>12,}")

        print("=" * 70)
        print(f"Total de parâmetros treináveis: {total:,}")
        print("=" * 70)

    def __repr__(self):
        layers_str = "\n  ".join(repr(l) for l in self.layers)
        return f"Sequential([\n  {layers_str}\n])"