"""
Character-level RNN: aprende a prever o próximo caractere.

Treina SimpleRNN e LSTM do zero, em NumPy puro, e gera texto novo
com amostragem por temperatura (temperature=0 → argmax; > 1 → mais
diverso). Usa gradient clipping para evitar explosão no BPTT.
"""
import os
import numpy as np

from nn import (
    Sequential, SimpleRNN, LSTM, Dense, ActivationLayer,
    Adam, CosineAnnealing, EarlyStopping, ModelCheckpoint,
)
from nn.utils import train_test_split


TEXT = (
    "the cat sat on the mat. "
    "the dog sat on the log. "
    "the cat chased the rat. "
    "the dog chased the cat. "
) * 4


def build_dataset(text, seq_len):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    V = len(chars)

    X_idx, y_idx = [], []
    for i in range(len(text) - seq_len):
        seq = text[i:i + seq_len]
        target = text[i + seq_len]
        X_idx.append([c2i[c] for c in seq])
        y_idx.append(c2i[target])

    X_idx = np.array(X_idx)
    y_idx = np.array(y_idx)

    X_oh = np.zeros((len(X_idx), seq_len, V))
    for i, seq in enumerate(X_idx):
        X_oh[i, np.arange(seq_len), seq] = 1.0

    y_oh = np.zeros((len(y_idx), V))
    y_oh[np.arange(len(y_idx)), y_idx] = 1.0

    return X_oh, y_oh, c2i, i2c, V


def _sample(probs, temperature, rng):
    """Amostra índice de `probs` com temperatura."""
    if temperature <= 0:
        return int(np.argmax(probs))
    logits = np.log(probs + 1e-12) / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def generate(model, seed_text, length, seq_len, c2i, i2c, V,
             temperature=0.8, seed=0):
    rng = np.random.default_rng(seed)
    result = seed_text
    for _ in range(length):
        window = result[-seq_len:]
        x = np.zeros((1, seq_len, V))
        for t, c in enumerate(window):
            x[0, t, c2i[c]] = 1.0
        probs = model.predict(x)[0]
        idx = _sample(probs, temperature, rng)
        result += i2c[idx]
    return result


def build_model(rnn_cls, V, hidden, seed):
    model = Sequential()
    model.add(rnn_cls(input_dim=V, hidden_dim=hidden,
                      return_sequences=False, init="xavier", seed=seed))
    model.add(ActivationLayer("tanh"))
    model.add(Dense(hidden, V, init="xavier"))
    model.add(ActivationLayer("softmax"))
    return model


def train_model(model, Xtr, ytr, Xval, yval, epochs, ckpt_path):
    # clip_norm evita explosão no BPTT sem atrapalhar convergência
    opt = Adam(lr=5e-3, clip_norm=5.0)
    model.compile(loss="cce", optimizer=opt)
    model.fit(
        Xtr, ytr,
        epochs=epochs, batch_size=32,
        validation_data=(Xval, yval),
        scheduler=CosineAnnealing(opt, t_max=epochs, eta_min=1e-4),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15, mode="min"),
            ModelCheckpoint(ckpt_path, monitor="val_loss", mode="min", verbose=False),
        ],
    )
    model.load(ckpt_path)


def main():
    SEQ_LEN = 8
    HIDDEN = 32
    EPOCHS = 100

    X, y, c2i, i2c, V = build_dataset(TEXT, SEQ_LEN)
    print(f"Vocabulário: {V} chars | Amostras: {len(X)} | seq_len: {SEQ_LEN}")

    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.1)

    os.makedirs("checkpoints", exist_ok=True)
    seeds = ["the cat ", "the dog ", "the rat "]

    for name, rnn_cls in [("SimpleRNN", SimpleRNN), ("LSTM", LSTM)]:
        print(f"\n=== {name} ===")
        model = build_model(rnn_cls, V, HIDDEN, seed=42)
        train_model(
            model, Xtr, ytr, Xval, yval,
            epochs=EPOCHS,
            ckpt_path=f"checkpoints/char_{name.lower()}_best.npz",
        )

        print(f"\nGeração com {name} (temperature=0.5):")
        for s in seeds:
            out = generate(model, s, 60, SEQ_LEN, c2i, i2c, V,
                           temperature=0.5, seed=0)
            print(f"  seed={s!r}")
            print(f"    → {out!r}")

        print(f"\nGeração com {name} (temperature=0.9 — mais criativa):")
        for s in seeds[:1]:
            out = generate(model, s, 60, SEQ_LEN, c2i, i2c, V,
                           temperature=0.9, seed=0)
            print(f"  seed={s!r}")
            print(f"    → {out!r}")


if __name__ == "__main__":
    main()