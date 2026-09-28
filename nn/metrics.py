import numpy as np


def accuracy(y_true, y_pred):
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    else:
        y_pred = (y_pred > 0.5).astype(int).ravel()
        y_true = y_true.ravel()
    return np.mean(y_pred == y_true)


def confusion_matrix(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[np.where(labels == t)[0][0], np.where(labels == p)[0][0]] += 1
    return cm, labels


def precision_recall_f1(y_true, y_pred, average="macro"):
    cm, _ = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp == 0, 0, tp / (tp + fp))
        recall = np.where(tp + fn == 0, 0, tp / (tp + fn))
        f1 = np.where(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
    if average == "macro":
        return precision.mean(), recall.mean(), f1.mean()
    return precision, recall, f1
