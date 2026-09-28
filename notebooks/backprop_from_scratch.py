# %% [markdown]
# # Backpropagation from Scratch
#
# **Como uma rede neural aprende, célula a célula.**
#
# Este notebook deriva o backpropagation à mão — sem `autograd`, sem
# PyTorch, sem TensorFlow — e prova que a derivação está correta comparando
# com a derivada numérica por diferenças finitas.
#
# Ao final, você terá:
#
# 1. Derivado o gradiente de uma camada densa com a regra da cadeia
# 2. Implementado forward e backward manualmente, em NumPy puro
# 3. Verificado numericamente que o gradiente está certo
# 4. Treinado uma rede no XOR do zero
# 5. Comparado com a implementação do pacote `nn`
#
# ---

# %% [markdown]
# ## 1. Setup

# %%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 100

# %% [markdown]
# ## 2. Um único neurônio, uma única amostra
#
# A unidade mais simples possível: uma entrada $x$, um peso $w$, um viés $b$,
# uma ativação sigmoide $\sigma$ e uma perda quadrática.
#
# **Forward:**
#
# $$
# z = w \cdot x + b
# \qquad
# a = \sigma(z)
# \qquad
# L = \tfrac{1}{2}(a - y)^2
# $$
#
# **Objetivo:** calcular $\dfrac{\partial L}{\partial w}$ e $\dfrac{\partial L}{\partial b}$.

# %%
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Forward
x, y = 0.8, 1.0
w, b = 0.5, -0.2

z = w * x + b
a = sigmoid(z)
L = 0.5 * (a - y) ** 2

print(f"z = {z:.4f}")
print(f"a = {a:.4f}")
print(f"L = {L:.4f}")

# %% [markdown]
# ## 3. Backward, à mão
#
# Regra da cadeia, um passo de cada vez:
#
# $$
# \frac{\partial L}{\partial a} = a - y
# \qquad
# \frac{\partial a}{\partial z} = a(1 - a)
# \qquad
# \frac{\partial z}{\partial w} = x
# \qquad
# \frac{\partial z}{\partial b} = 1
# $$
#
# Portanto:
#
# $$
# \frac{\partial L}{\partial w}
# = (a - y) \cdot a(1 - a) \cdot x
# \qquad
# \frac{\partial L}{\partial b}
# = (a - y) \cdot a(1 - a)
# $$

# %%
# Backward analítico
dL_da = a - y
da_dz = a * (1 - a)
dz_dw = x
dz_db = 1.0

dL_dw = dL_da * da_dz * dz_dw
dL_db = dL_da * da_dz * dz_db

print(f"dL/dw = {dL_dw:.6f}")
print(f"dL/db = {dL_db:.6f}")

# %% [markdown]
# ## 4. Verificação numérica
#
# A derivada numérica (diferenças centrais) é:
#
# $$
# \frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon}
# $$
#
# Se o gradiente analítico bater com este, nossa derivação está correta.
# Esse é o teste que pega bugs que o treino sozinho esconde.

# %%
def L_of(w, b, x, y):
    z = w * x + b
    a = sigmoid(z)
    return 0.5 * (a - y) ** 2

eps = 1e-6

# Numérico para w
L_plus  = L_of(w + eps, b, x, y)
L_minus = L_of(w - eps, b, x, y)
dL_dw_num = (L_plus - L_minus) / (2 * eps)

# Numérico para b
L_plus  = L_of(w, b + eps, x, y)
L_minus = L_of(w, b - eps, x, y)
dL_db_num = (L_plus - L_minus) / (2 * eps)

print(f"dL/dw  analítico = {dL_dw:.8f}   numérico = {dL_dw_num:.8f}")
print(f"dL/db  analítico = {dL_db:.8f}   numérico = {dL_db_num:.8f}")
print()
print("Erro relativo w:", abs(dL_dw - dL_dw_num) / (abs(dL_dw) + 1e-12))
print("Erro relativo b:", abs(dL_db - dL_db_num) / (abs(dL_db) + 1e-12))

# %% [markdown]
# ## 5. Escalando: uma camada densa
#
# Agora com mini-batch $X \in \mathbb{R}^{m \times d_{in}}$,
# pesos $W \in \mathbb{R}^{d_{in} \times d_{out}}$,
# viés $b \in \mathbb{R}^{1 \times d_{out}}$, ativação $f$ e perda $L$.
#
# $$
# Z = XW + b
# \qquad
# A = f(Z)
# $$
#
# Seja $dZ := \dfrac{\partial L}{\partial Z}$ a "mensagem" que chega da
# camada seguinte. Então:
#
# $$
# \boxed{
# \begin{aligned}
# \frac{\partial L}{\partial W} &= \frac{1}{m} X^\top \cdot dZ \\[4pt]
# \frac{\partial L}{\partial b} &= \frac{1}{m} \sum_{i} dZ_i \\[4pt]
# \frac{\partial L}{\partial X} &= dZ \cdot W^\top
# \end{aligned}}
# $$
#
# Note o `/m` nos **parâmetros** (para o otimizador receber o gradiente da
# loss *média*) e a **ausência** de `/m` no `dX` (para não dividir duas
# vezes ao longo da cadeia).

# %%
class DenseManual:
    """Uma camada densa + sigmoid, com forward/backward manuais."""

    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(1.0 / in_dim)
        self.b = np.zeros((1, out_dim))
        self._X = None
        self._Z = None
        self._A = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self._X = X
        self._Z = X @ self.W + self.b
        self._A = sigmoid(self._Z)
        return self._A

    def backward(self, dA):
        # ∂L/∂Z = ∂L/∂A ⊙ σ'(Z)
        dZ = dA * self._A * (1 - self._A)

        m = self._X.shape[0]
        self.dW = (self._X.T @ dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = dZ @ self.W.T

        # Guarda dZ para quem quiser inspecionar
        self._dZ = dZ
        return dX


# %% [markdown]
# ## 6. Uma rede de duas camadas
#
# Empilhamos duas `DenseManual` e adicionamos o gradiente da loss
# quadrática no topo:
#
# $$
# L = \tfrac{1}{2m} \sum_i (A_i - y_i)^2
# \qquad
# \frac{\partial L}{\partial A} = \frac{A - y}{m}
# $$

# %%
def forward_full(model, X):
    A1 = model["fc1"].forward(X)
    A2 = model["fc2"].forward(A1)
    return A2


def backward_full(model, X, y, y_pred):
    m = X.shape[0]

    # dL/dA2
    dA2 = (y_pred - y) / m

    # Desce pela segunda camada
    dA1 = model["fc2"].backward(dA2)

    # Desce pela primeira
    model["fc1"].backward(dA1)


def loss_mse(y_pred, y):
    return 0.5 * np.mean((y_pred - y) ** 2)


# %% [markdown]
# ## 7. Verificação numérica na rede inteira
#
# O mesmo teste de antes, mas agora em cada parâmetro da rede.

# %%
# Dados pequenos, um problema de classificação binária
rng = np.random.default_rng(0)
X = rng.standard_normal((8, 3))
y = (rng.random((8, 1)) > 0.5).astype(float)

model = {
    "fc1": DenseManual(3, 4),
    "fc2": DenseManual(4, 1),
}

# Forward + backward
y_pred = forward_full(model, X)
backward_full(model, X, y, y_pred)


def loss_of(model):
    yp = forward_full(model, X)
    return loss_mse(yp, y)


def numerical_grad(model, layer, key, eps=1e-6):
    p = getattr(model[layer], key)
    grad = np.zeros_like(p)
    it = np.nditer(p, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = p[idx]

        p[idx] = orig + eps
        Lp = loss_of(model)
        p[idx] = orig - eps
        Lm = loss_of(model)
        p[idx] = orig

        grad[idx] = (Lp - Lm) / (2 * eps)
        it.iternext()
    return grad


print(f"{'Layer':<8}{'Param':<8}{'Analítico':<18}{'Numérico':<18}{'Erro rel':<12}")
print("-" * 64)

for layer in ("fc1", "fc2"):
    for key in ("W", "b"):
        analytic = getattr(model[layer], "d" + key)
        numeric = numerical_grad(model, layer, key)
        err = np.linalg.norm(analytic - numeric) / (
            np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12
        )
        print(f"{layer:<8}{key:<8}{analytic.ravel()[0]:<18.6f}"
              f"{numeric.ravel()[0]:<18.6f}{err:<12.2e}")

# %% [markdown]
# Se todos os erros relativos forem < `1e-6`, o backprop manual está correto.
# Essa é a **mesma** verificação que a suíte de testes do pacote `nn` roda
# em `tests/test_gradients.py` e `tests/test_conv.py`.

# %% [markdown]
# ## 8. Treinando XOR com o modelo manual
#
# O XOR é o problema canônico que **exige** uma camada escondida.
# Uma regressão logística (uma camada) não consegue separar as 4 amostras.

# %%
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([[0], [1], [1], [0]], dtype=float)

model = {
    "fc1": DenseManual(2, 8),
    "fc2": DenseManual(8, 1),
}

lr = 0.5
history = []

for epoch in range(5000):
    y_pred = forward_full(model, X_xor)
    backward_full(model, X_xor, y_xor, y_pred)

    for layer in ("fc1", "fc2"):
        model[layer].W -= lr * model[layer].dW
        model[layer].b -= lr * model[layer].db

    if epoch % 500 == 0:
        loss = loss_mse(y_pred, y_xor)
        history.append((epoch, loss))

for epoch, loss in history:
    print(f"Epoch {epoch:5d} | loss = {loss:.6f}")

print()
print("Predições finais:")
print(np.round(forward_full(model, X_xor), 3).ravel())

# %% [markdown]
# ## 9. Visualizando a curva de loss

# %%
losses = []
for epoch in range(5000):
    y_pred = forward_full(model, X_xor)
    backward_full(model, X_xor, y_xor, y_pred)
    for layer in ("fc1", "fc2"):
        model[layer].W -= lr * model[layer].dW
        model[layer].b -= lr * model[layer].db
    losses.append(loss_mse(y_pred, y_xor))

plt.plot(losses, linewidth=0.8)
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("XOR — modelo manual")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Mesma coisa com o pacote `nn`
#
# A rede `2 → 8 → 1` treinada acima é **exatamente** o que o `nn.Sequential`
# do pacote faz por baixo. A diferença é que o pacote generaliza para
# convolução, batch norm, dropout, Adam, cosine annealing, etc.
#
# Vamos treinar a mesma rede com o pacote e comparar.

# %%
from nn import Sequential, Dense, ActivationLayer, Adam
from nn.utils import evaluate_and_report

model_pkg = Sequential()
model_pkg.add(Dense(2, 8, init="xavier"))
model_pkg.add(ActivationLayer("tanh"))
model_pkg.add(Dense(8, 1, init="xavier"))
model_pkg.add(ActivationLayer("sigmoid"))

model_pkg.compile(loss="bce", optimizer=Adam(lr=0.05))
model_pkg.fit(X_xor, y_xor, epochs=200, batch_size=4, verbose=False)

print("Predições com nn.Sequential:")
print(np.round(model_pkg.predict(X_xor), 3).ravel())

# %% [markdown]
# ## 11. Conclusão
#
# Você acabou de:
#
# 1. Derivar $\partial L/\partial W$ e $\partial L/\partial b$ à mão
# 2. Implementar forward e backward em NumPy puro
# 3. **Provar** que o gradiente está correto com diferenças finitas
# 4. Treinar no XOR do zero
# 5. Comparar com a implementação do pacote `nn`
#
# O resto do pacote `nn` é esta mesma ideia, generalizada:
#
# - `Conv2D` → troca a multiplicação matricial por uma convolução via `im2col`
# - `BatchNorm` → insere uma operação de normalização entre camadas
# - `Dropout` → mascara ativações aleatoriamente durante o treino
# - `Adam` → troca o SGD pela regra de Adam (média móvel + RMS)
# - `Softmax + CCE` → funde duas operações em uma para estabilidade numérica
#
# Em todos os casos, a regra é a mesma:
#
# > **Derivar à mão, implementar, verificar numericamente, treinar.**
#
# Essa disciplina é o que separa "usar uma biblioteca" de "entender o que
# a biblioteca faz". E é o que este projeto tenta ensinar.
