"""
Visualizações para treino e inspeção de modelos.

Usa matplotlib com backend Agg (sem display) para funcionar em CI/headless.
Todas as funções retornam a `Figure` e, se `path` for dado, salvam em disco.
"""

import numpy as np


def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib é necessário para visualização. "
            "Instale com: pip install matplotlib"
        ) from e


def plot_loss_curve(history, path=None, title="Training curve",
                    log_scale=False):
    """Plota loss de treino (e validação, se houver) por época."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = np.arange(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"], label="train loss", marker="o", markersize=3)

    if "val_loss" in history and len(history["val_loss"]) > 0:
        ax.plot(epochs, history["val_loss"], label="val loss",
                marker="o", markersize=3)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=120)
        print(f"Figura salva em {path}")

    return fig


def plot_lr_curve(history, path=None, title="Learning rate schedule"):
    """Plota o LR por época (se o `history` tiver a chave 'lr')."""
    plt = _get_plt()
    if "lr" not in history or len(history["lr"]) == 0:
        raise ValueError("history não contém a chave 'lr'")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(history["lr"]) + 1), history["lr"],
            marker="o", markersize=3, color="C2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=120)
        print(f"Figura salva em {path}")

    return fig


def plot_confusion_matrix(cm, labels=None, path=None,
                          title="Confusion matrix", cmap="Blues"):
    """Plota matriz de confusão com anotações numéricas."""
    plt = _get_plt()
    cm = np.asarray(cm)
    if labels is None:
        labels = list(range(cm.shape[0]))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=cmap)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=120)
        print(f"Figura salva em {path}")

    return fig


def plot_filters(conv_layer, path=None, max_filters=None,
                 title="Conv2D filters"):
    """
    Visualiza os filtros aprendidos de uma `Conv2D`.

    Como cada filtro tem shape (C_in, kH, kW), mostra a média sobre C_in
    para dar uma imagem única 2D.
    """
    plt = _get_plt()
    W = conv_layer.W
    C_out, C_in, kH, kW = W.shape
    if max_filters is not None:
        C_out = min(C_out, max_filters)

    cols = int(np.ceil(np.sqrt(C_out)))
    rows = int(np.ceil(C_out / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i >= C_out:
            continue
        filt = W[i].mean(axis=0)  # média sobre canais de entrada
        ax.imshow(filt, cmap="gray")
        ax.set_title(f"#{i}", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=120)
        print(f"Figura salva em {path}")

    return fig
