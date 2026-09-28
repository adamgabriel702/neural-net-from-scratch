"""
MLP no MNIST. Usa `nn.datasets` e `nn.visualization` para não duplicar código.
"""
import os
import numpy as np

from nn import (
    Sequential, Dense, BatchNorm, ActivationLayer, Dropout,
    Adam, CosineAnnealing, EarlyStopping, ModelCheckpoint,
)
from nn.datasets import load_mnist
from nn.utils import one_hot, train_test_split, evaluate_and_report
from nn.visualization import (
    plot_loss_curve, plot_confusion_matrix,
)


def build_model():
    model = Sequential()
    model.add(Dense(784, 128, init="he"))
    model.add(BatchNorm(128))
    model.add(ActivationLayer("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, 64, init="he"))
    model.add(BatchNorm(64))
    model.add(ActivationLayer("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, 10, init="xavier"))
    model.add(ActivationLayer("softmax"))
    return model


def main():
    N = 10000
    EPOCHS = 20

    Xtr, ytr, Xte, yte = load_mnist(flatten=True)
    Xtr, ytr = Xtr[:N], ytr[:N]

    ytr_oh = one_hot(ytr, 10)
    yte_oh = one_hot(yte, 10)

    Xtr, Xval, ytr_oh, yval_oh = train_test_split(Xtr, ytr_oh, test_size=0.1)

    model = build_model()
    opt = Adam(lr=1e-3)
    model.compile(loss="cce", optimizer=opt)
    model.summary()

    scheduler = CosineAnnealing(opt, t_max=EPOCHS, eta_min=1e-5)

    os.makedirs("checkpoints", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, mode="min"),
        ModelCheckpoint("checkpoints/mnist_mlp_best.npz",
                        monitor="val_loss", mode="min"),
    ]

    model.fit(
        Xtr, ytr_oh,
        epochs=EPOCHS, batch_size=64,
        validation_data=(Xval, yval_oh),
        scheduler=scheduler,
        callbacks=callbacks,
    )

    os.makedirs("figures", exist_ok=True)
    plot_loss_curve(model.history, path="figures/mnist_mlp_loss.png",
                    title="MNIST MLP — training curve")

    model.load("checkpoints/mnist_mlp_best.npz")
    report = evaluate_and_report(model, Xte, yte, yte_oh, title="MNIST MLP — Test")
    plot_confusion_matrix(report["confusion_matrix"],
                          path="figures/mnist_mlp_confusion.png",
                          title="MNIST MLP — Confusion matrix")

    model.save("checkpoints/mnist_mlp_final.npz")


if __name__ == "__main__":
    main()