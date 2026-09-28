"""
CNN no MNIST. Reusa os helpers compartilhados e plota os filtros aprendidos.
"""
import os
import numpy as np

from nn import (
    Sequential, Conv2D, BatchNorm2D, MaxPool2D, Flatten, Dense,
    ActivationLayer, Dropout, Adam, CosineAnnealing,
    EarlyStopping, ModelCheckpoint, DataLoader,
)
from nn.datasets import load_mnist
from nn.utils import one_hot, train_test_split, evaluate_and_report
from nn.visualization import (
    plot_loss_curve, plot_lr_curve, plot_confusion_matrix, plot_filters,
)


def build_model():
    model = Sequential()
    model.add(Conv2D(1, 8, kernel_size=3, stride=1, padding=1, init="he"))
    model.add(BatchNorm2D(8))
    model.add(ActivationLayer("relu"))
    model.add(MaxPool2D(2))
    model.add(Conv2D(8, 16, kernel_size=3, stride=1, padding=1, init="he"))
    model.add(BatchNorm2D(16))
    model.add(ActivationLayer("relu"))
    model.add(MaxPool2D(2))
    model.add(Flatten())
    model.add(Dense(16 * 7 * 7, 64, init="he"))
    model.add(ActivationLayer("relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, 10, init="xavier"))
    model.add(ActivationLayer("softmax"))
    return model


def main():
    import os
    from nn import DataLoader, Compose, RandomShift, GaussianNoise

    smoke = os.environ.get("SMOKE") == "1"
    N = 512 if smoke else 8000
    EPOCHS = 1 if smoke else 15

    Xtr, ytr, Xte, yte = load_mnist(flatten=False)
    Xtr, ytr = Xtr[:N], ytr[:N]

    ytr_oh = one_hot(ytr, 10)
    yte_oh = one_hot(yte, 10)

    Xtr, Xval, ytr_oh, yval_oh = train_test_split(Xtr, ytr_oh, test_size=0.1)

    model = build_model()
    opt = Adam(lr=2e-3)
    model.compile(loss="cce", optimizer=opt)
    if not smoke:
        model.summary()

    scheduler = CosineAnnealing(opt, t_max=EPOCHS, eta_min=1e-5)

    # Data augmentation: pequeno deslocamento + ruído leve
    # (flip horizontal NÃO é usado: 6 e 9 se confundem)
    aug = Compose([
        RandomShift(max_shift=2, seed=0),
        GaussianNoise(sigma=0.05, seed=0),
    ])

    train_loader = DataLoader(
        Xtr, ytr_oh,
        batch_size=32,
        shuffle=True,
        transform=aug,
        seed=0,
    )

    os.makedirs("checkpoints", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, mode="min"),
        ModelCheckpoint("checkpoints/mnist_cnn_best.npz",
                        monitor="val_loss", mode="min", verbose=not smoke),
    ]

    model.fit(
        train_loader,
        epochs=EPOCHS,
        verbose=not smoke,
        validation_data=(Xval, yval_oh),
        scheduler=scheduler,
        callbacks=callbacks,
    )

    if not smoke:
        os.makedirs("figures", exist_ok=True)
        plot_loss_curve(model.history, path="figures/mnist_cnn_loss.png",
                        title="MNIST CNN — training curve")
        if "lr" in model.history:
            plot_lr_curve(model.history, path="figures/mnist_cnn_lr.png")

    model.load("checkpoints/mnist_cnn_best.npz")
    report = evaluate_and_report(model, Xte, yte, yte_oh,
                                 title="MNIST CNN — Test",
                                 print_cm=not smoke)

    if not smoke:
        plot_confusion_matrix(report["confusion_matrix"],
                              path="figures/mnist_cnn_confusion.png",
                              title="MNIST CNN — Confusion matrix")
        plot_filters(model.layers[0], path="figures/mnist_cnn_filters.png",
                     title="Filtros da 1ª Conv2D (média sobre canais)")

    model.save("checkpoints/mnist_cnn_final.npz")

if __name__ == "__main__":
    main()