"""
Callbacks executados ao final de cada época.

Interface mínima: `on_epoch_end(model, epoch, logs)`.
`logs` é um dicionário com 'loss' e, se houver, 'val_loss'.
"""

import os
import numpy as np


class Callback:
    def on_train_begin(self, model, logs=None): pass
    def on_epoch_end(self, model, epoch, logs): pass
    def on_train_end(self, model, logs=None): pass


class EarlyStopping(Callback):
    """
    Interrompe o treino quando `monitor` para de melhorar.

    mode='min' (default) para losses; use mode='max' para métricas
    como accuracy.
    """

    def __init__(self, monitor="val_loss", patience=5, min_delta=0.0, mode="min"):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.wait = 0
        self.stopped_epoch = None
        self.should_stop = False

    def _is_improvement(self, current):
        if self.best is None:
            return True
        if self.mode == "min":
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta

    def on_epoch_end(self, model, epoch, logs):
        if self.monitor not in logs:
            return
        current = logs[self.monitor]

        if self._is_improvement(current):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch


class ModelCheckpoint(Callback):
    """
    Salva o melhor modelo em disco conforme `monitor`.

    Reutiliza `model.save(path)` — mesmo formato `.npz`.
    """

    def __init__(self, filepath, monitor="val_loss", mode="min", verbose=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = None

    def _is_improvement(self, current):
        if self.best is None:
            return True
        if self.mode == "min":
            return current < self.best
        return current > self.best

    def on_epoch_end(self, model, epoch, logs):
        if self.monitor not in logs:
            return
        current = logs[self.monitor]
        if self._is_improvement(current):
            self.best = current
            model.save(self.filepath)
            if self.verbose:
                print(f"  [checkpoint] {self.monitor}={current:.6f} salvo em {self.filepath}")


class History(Callback):
    """
    Acumula histórico em `model.history` quando o `fit` é chamado
    várias vezes em pedaços (chamadas separadas de fit(epochs=1)).
    """

    def __init__(self):
        self.history = {"loss": [], "val_loss": [], "lr": []}

    def on_epoch_end(self, model, epoch, logs):
        self.history["loss"].append(logs.get("loss"))
        if "val_loss" in logs:
            self.history["val_loss"].append(logs["val_loss"])
        if model.optimizer is not None:
            self.history["lr"].append(model.optimizer.lr)
