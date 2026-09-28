# examples/xor.py
import numpy as np
from nn import Sequential, Dense, ActivationLayer, Adam, metrics

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

model = Sequential()
model.add(Dense(2, 8, init="xavier"))
model.add(ActivationLayer("tanh"))
model.add(Dense(8, 1, init="xavier"))
model.add(ActivationLayer("sigmoid"))

model.compile(loss="bce", optimizer=Adam(lr=0.01))
model.fit(X, y, epochs=200, batch_size=4)

preds = model.predict(X)
print("Predições:", np.round(preds, 3).ravel())
print("Accuracy:", metrics.accuracy(y, preds))
