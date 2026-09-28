"""
Wrappers que aplicam uma camada sobre uma dimensão específica.

TimeDistributed aplica uma camada por timestep de uma sequência:
    RNN(return_seq=True) → TimeDistributed(Dense) → softmax por step

Útil para:
- Sequência → sequência (ex.: POS tagging, NER)
- Empilhar RNNs com FC no meio
- Aplicar Dense sobre imagens (N, C, H, W) como se fossem H*W features
  separadas
"""

import numpy as np

from .layers import Layer


class TimeDistributed(Layer):
    """
    Aplica uma camada a cada timestep de uma sequência.

    Entrada:  (N, T, ...) — qualquer shape por timestep
    Saída:    (N, T, ...) — mesmo shape, transformado pelo layer interno

    Uso:
        TimeDistributed(Dense(32, 10))  # (N, T, 32) → (N, T, 10)

    Os parâmetros e buffers (se houver, como em BatchNorm) são
    delegados ao layer interno. `params()`, `grads()` e `buffers()`
    repassam direto.
    """

    def __init__(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError(
                f"TimeDistributed espera um Layer, recebeu {type(layer).__name__}"
            )
        self.layer = layer
        self._input_shape = None
        self._output_shape = None

    def forward(self, X, training=True):
        if X.ndim < 3:
            raise ValueError(
                f"TimeDistributed espera (N, T, ...), recebeu shape {X.shape}"
            )
        self._input_shape = X.shape
        N, T = X.shape[0], X.shape[1]
        rest = X.shape[2:]

        # Achata batch e tempo: (N*T, ...)
        X_flat = X.reshape(N * T, *rest)
        out_flat = self.layer.forward(X_flat, training=training)

        self._output_shape = out_flat.shape
        return out_flat.reshape(N, T, *out_flat.shape[1:])

    def backward(self, dout):
        N, T = dout.shape[0], dout.shape[1]
        dout_flat = dout.reshape(N * T, *dout.shape[2:])
        dX_flat = self.layer.backward(dout_flat)
        return dX_flat.reshape(self._input_shape)

    def params(self):
        return self.layer.params()

    def grads(self):
        return self.layer.grads()

    def buffers(self):
        return self.layer.buffers()

    def __repr__(self):
        return f"TimeDistributed({self.layer!r})"
