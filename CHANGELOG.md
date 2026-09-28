# Changelog

Todas as mudanças notáveis deste projeto são documentadas neste arquivo.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o versionamento segue [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Unreleased]

## [0.2.0] — 2026-09-28

### Adicionado
- **Camadas recorrentes**: `SimpleRNN` e `LSTM` com BPTT manual
  - `return_sequences=True/False` para empilhar RNNs ou passar para `Dense`
  - Forget bias inicializado em 1.0 (prática comum do paper original)
- **Wrapper `TimeDistributed`**: aplica um `Layer` por timestep
  - Permite `RNN(return_seq=True) → TimeDistributed(Dense) → softmax`
- **Gradient clipping** (`clip_norm`) em `SGD`, `Momentum` e `Adam`
  - Função pública `clip_gradients` reutilizável em loops manuais
- **Exemplo `examples/char_rnn.py`**: geração de texto caractere a caractere
  com amostragem por temperatura
- 21 novos testes: `test_recurrent.py` (14) + `test_wrappers.py` (7)
- 7 novos testes de gradient clipping: `test_gradient_clipping.py`

### Mudado
- `nn/optimizers.py`: classe base `Optimizer` com `_clip` compartilhado
- `nn/__init__.py`: exporta `clip_gradients`, `TimeDistributed`, `SimpleRNN`, `LSTM`

## [0.1.0] — 2026-09-28

### Adicionado
- **Camadas**: `Dense`, `ActivationLayer`, `Dropout`, `BatchNorm`,
  `BatchNorm2D`, `Conv2D`, `MaxPool2D`, `Flatten`
- **Ativações**: Sigmoid, ReLU, Tanh, Softmax, Linear
- **Losses**: MSE, Binary Cross-Entropy, Categorical Cross-Entropy
  (com fusão Softmax + CCE)
- **Otimizadores**: SGD, Momentum, Adam (com bias correction)
- **Schedulers**: `StepLR`, `ExponentialLR`, `CosineAnnealing`, `WarmupCosine`
- **Callbacks**: `EarlyStopping`, `ModelCheckpoint`, `History`
- **DataLoader** com batching, shuffle e `transform`
- **Data augmentation**: `Compose`, `RandomHorizontalFlip`, `RandomShift`,
  `RandomRotation90`, `GaussianNoise`
- **Métricas**: accuracy, confusion matrix, precision / recall / F1
- **Dataset MNIST** com download automático
- **Visualizações** com matplotlib: loss curves, confusion matrix, filtros
- **Utils**: `train_test_split`, `standardize`, `one_hot`, `batch_iterator`
- **68 testes**, incluindo verificação de gradiente numérico contra
  backprop analítico para `Dense`, `Conv2D`, `BatchNorm` e `BatchNorm2D`
- **CI** no GitHub Actions (Python 3.10, 3.11, 3.12)
- **Notebook didático** (`notebooks/backprop_from_scratch.ipynb`)
- **Exemplos**: XOR, Iris, MNIST (MLP), MNIST (CNN, 98.5% com augmentation)
- Empacotamento com `pyproject.toml`, `MANIFEST.in` e PEP 561 (`py.typed`)

[Unreleased]: https://github.com/adamgabriel702/neural-net-from-scratch/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/adamgabriel702/neural-net-from-scratch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/adamgabriel702/neural-net-from-scratch/releases/tag/v0.1.0
