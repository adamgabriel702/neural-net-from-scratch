"""
Helpers compartilhados pelos testes.

O mais importante aqui é `numerical_grad` / `check_gradients`: a
verificação de gradiente numérico é o pilar que garante que o backprop
analítico de cada camada está correto.
"""

import numpy as np


def numerical_grad(model, X, y, layer_idx, key, training=False, eps=1e-6):
    """
    Gradiente numérico (diferença central) para um parâmetro específico.

    Usa `model.forward(training=training)` para respeitar modos de
    BatchNorm/Dropout. Para camadas com BatchNorm treinável, use
    `training=True` para bater com o forward que gerou os gradientes
    analíticos.
    """
    p = model.layers[layer_idx].params()[key]
    grad = np.zeros_like(p)

    it = np.nditer(p, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = p[idx]

        p[idx] = orig + eps
        loss_plus = model.loss_fn(y, model.forward(X, training=training))

        p[idx] = orig - eps
        loss_minus = model.loss_fn(y, model.forward(X, training=training))

        p[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()

    return grad


def check_gradients(model, X, y, layer_indices=None, keys=None,
                    atol=1e-5, training=False, verbose=False):
    """
    Compara gradiente analítico com numérico para os parâmetros dos
    layers indicados. Levanta `AssertionError` se algum divergir.

    Parâmetros
    ----------
    layer_indices : iterable de int ou None
        Layers a checar. `None` = todos os layers com parâmetros.
    keys : iterable de str ou None
        Chaves de parâmetros (ex.: {"W", "b"}). `None` = todas.
    atol : float
        Tolerância do erro relativo.
    training : bool
        Passa para `forward()`. `True` para BatchNorm/Dropout.
    verbose : bool
        Imprime o erro de cada parâmetro.
    """
    # Forward/backward para preencher os gradientes analíticos
    model.forward(X, training=training)
    model.backward(y)

    if layer_indices is None:
        layer_indices = range(len(model.layers))

    for i in layer_indices:
        layer = model.layers[i]
        params = layer.params()
        if not params:
            continue
        grads = layer.grads()
        keys_to_check = keys if keys is not None else params.keys()

        for key in keys_to_check:
            analytic = grads[key]
            numeric = numerical_grad(model, X, y, i, key, training=training)
            denom = np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12
            diff = np.linalg.norm(analytic - numeric) / denom

            if verbose:
                print(
                    f"  layer[{i}] {layer.__class__.__name__} {key}: "
                    f"diff = {diff:.2e}"
                )

            assert diff < atol, (
                f"Gradiente errado em layer {i} "
                f"({layer.__class__.__name__}) param {key}\n"
                f"erro relativo = {diff:.2e}"
            )
