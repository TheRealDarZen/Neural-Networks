from activation import *
from layers import *
from loss import *
from data_loader import *

# Variables
LR = 0.001
BATCH_SIZE = 16
EPOCHS = 60
HIDDEN_LAYER_SIZE = 64


def get_batches(X, y, batch_size=32):
    m = X.shape[1]
    indices = np.random.permutation(m)
    X_shuffled = X[:, indices]
    y_shuffled = y[:, indices]

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[:, i:i + batch_size]
        y_batch = y_shuffled[:, i:i + batch_size]
        yield X_batch, y_batch


def main():
    X_train, X_test, y_train, y_test = get_data()

    # Network
    layers = [
        Linear(X_train.shape[0], HIDDEN_LAYER_SIZE),
        ReLU(),
        Linear(HIDDEN_LAYER_SIZE, 5),
        Softmax()
    ]
    loss_fn = CategoricalCrossEntropy()

    # Train loop
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for X_batch, y_batch in get_batches(X_train, y_train, BATCH_SIZE):
            # Forward
            out = X_batch
            for layer in layers:
                out = layer.forward(out)
            y_pred = out

            # Loss
            loss = loss_fn.loss(y_pred, y_batch)
            epoch_loss += loss

            # Backward
            grad = loss_fn.grad()
            for layer in reversed(layers):
                grad = layer.backward(grad)

            # Update
            for layer in layers:
                if isinstance(layer, Linear):
                    layer.update(LR)

        # Avg epoch loss
        epoch_loss /= (X_train.shape[1] / BATCH_SIZE)
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {epoch_loss:.4f}")

    # Eval
    out = X_test
    for layer in layers:
        out = layer.forward(out)
    y_pred = out

    # Results
    pred_classes = np.argmax(y_pred, axis=0)
    acc = np.mean(pred_classes == y_test)
    print("Test accuracy:", acc)


if __name__ == "__main__":
    main()

