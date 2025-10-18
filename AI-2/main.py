import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Preprocessing

X = X.dropna()
y = y.loc[X.index]

cat_var_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

X = pd.get_dummies(X, columns=cat_var_cols, drop_first=True, dtype=int)

y.loc[:, "num"] = y["num"].apply(lambda x: 1 if x != 0 else 0)

print(X.head())
print(y.head())

# Split and standardization

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(y_true, y_pred):
    m = y_true.shape[0]
    cost = - (1/m) * np.sum(y_true*np.log(y_pred + 1e-8) + (1 - y_true)*np.log(1 - y_pred + 1e-8))
    return cost


def train(X, y, lr=0.001, epochs=10000, batch_size=16):
    m, n = X.shape
    W = np.random.randn(n, 1)
    b = 0.0

    costs = []
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_cost = 0
        num_batches = 0

        # Iterate through batches
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            batch_m = X_batch.shape[0]

            # Forward pass
            z = np.dot(X_batch, W) + b
            y_pred = sigmoid(z)

            # Cost
            batch_cost = compute_cost(y_batch, y_pred)
            epoch_cost += batch_cost
            num_batches += 1

            # Gradients
            dW = (1 / batch_m) * np.dot(X_batch.T, (y_pred - y_batch))
            db = (1 / batch_m) * np.sum(y_pred - y_batch)

            # Update weights
            W -= lr * dW
            b -= lr * db

        # Average cost for the epoch
        avg_epoch_cost = epoch_cost / num_batches
        costs.append(avg_epoch_cost)

    return W, b, costs


def predict(X, W, b, threshold=0.5):
    probs = sigmoid(np.dot(X, W) + b)
    return (probs >= threshold).astype(int)

# Training
W, b, costs = train(X_train, y_train)

# Eval
y_pred = predict(X_test, W, b)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1-score:", f1)
print("Last cost:", costs[-1])

plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost over time")
plt.show()
