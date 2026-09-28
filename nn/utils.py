"""
Utilitários de pré-processamento, split, métricas de relatório e batching.
"""

import numpy as np


# ============================================================
# Pré-processamento
# ============================================================
def train_test_split(X, y, test_size=0.2, seed=42, shuffle=True):
    """Divide (X, y) em treino/teste."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    cut = int(n * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


def standardize(X, mean=None, std=None, eps=1e-8):
    """
    Padroniza X para média 0 e desvio 1 por coluna.

    Se `mean`/`std` forem fornecidos, usa-os (para aplicar as estatísticas
    do treino no teste). Senão, calcula a partir de X.
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    return (X - mean) / (std + eps), mean, std


def one_hot(y, n_classes=None):
    """Converte y (N,) int para matriz (N, n_classes) float64."""
    y = np.asarray(y).astype(int).ravel()
    if n_classes is None:
        n_classes = y.max() + 1
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


# ============================================================
# Batching
# ============================================================
def batch_iterator(X, y, batch_size, shuffle=True, seed=None):
    """
    Gera mini-batches `(X_batch, y_batch)`.

    Útil quando você quer controle manual do loop de treino em vez de
    chamar `model.fit(...)`.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, batch_size):
        batch = idx[s:s + batch_size]
        yield X[batch], y[batch]


# ============================================================
# Relatório de métricas
# ============================================================
def evaluate_and_report(model, X, y, y_oh, title="Test", print_cm=True):
    """
    Avalia o modelo e imprime loss, accuracy e matriz de confusão.

    Parâmetros
    ----------
    model : Sequential
    X : array (N, ...)
    y : array (N,) int — rótulos originais (para confusion matrix)
    y_oh : array (N, C) one-hot
    title : str
    print_cm : bool

    Retorna
    -------
    dict com `loss`, `accuracy`, `confusion_matrix`, `labels`.
    """
    from . import metrics as M

    preds = model.predict(X)
    loss = model.evaluate(X, y_oh)
    acc = M.accuracy(y_oh, preds)

    if preds.ndim > 1 and preds.shape[1] > 1:
        pred_classes = np.argmax(preds, axis=1)
    else:
        pred_classes = (preds.ravel() > 0.5).astype(int)

    cm, labels = M.confusion_matrix(y, pred_classes)

    print(f"\n[{title}]")
    print(f"  Loss:     {loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    if print_cm:
        print("\n  Matriz de confusão:")
        header = "       " + " ".join(f"{l:>5d}" for l in labels)
        print(header)
        for i, row in enumerate(cm):
            print(f"   {labels[i]:>3d} " + " ".join(f"{v:>5d}" for v in row))

    try:
        p, r, f1 = M.precision_recall_f1(y, pred_classes)
        print(f"  Macro P/R/F1: {p:.4f} / {r:.4f} / {f1:.4f}")
    except Exception:
        pass

    return {
        "loss": loss,
        "accuracy": acc,
        "confusion_matrix": cm,
        "labels": labels,
        "predictions": preds,
    }