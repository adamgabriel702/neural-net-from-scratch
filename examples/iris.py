# examples/iris.py
from sklearn.datasets import load_iris
from nn import Sequential, Dense, ActivationLayer, Adam, metrics, utils

X, y = load_iris(return_X_y=True)
X, _, _ = utils.standardize(X)
y_oh = utils.one_hot(y, 3)

Xtr, Xte, ytr, yte = utils.train_test_split(X, y_oh, test_size=0.2)

model = Sequential()
model.add(Dense(4, 16, init="he"))
model.add(ActivationLayer("relu"))
model.add(Dense(16, 8, init="he"))
model.add(ActivationLayer("relu"))
model.add(Dense(8, 3, init="xavier"))
model.add(ActivationLayer("softmax"))

model.compile(loss="cce", optimizer=Adam(lr=0.01))
model.fit(Xtr, ytr, epochs=300, batch_size=16)

preds = model.predict(Xte)
print("Accuracy teste:", metrics.accuracy(yte, preds))
