import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_imdb(vocab_size=10000):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    return x_train, y_train, x_test, y_test


def pad_batch(batch, max_len):
    sequences, labels = zip(*batch)
    seq_tensors = [torch.tensor(seq[:max_len]) for seq in sequences]
    seq_padded = pad_sequence(seq_tensors, batch_first=True, padding_value=0)
    return seq_padded, torch.tensor(labels)


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, rnn_type="RNN"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.rnn(x)
        last = out[:, -1, :]
        return torch.sigmoid(self.fc(last))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad()
        preds = model(x).squeeze()
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += ((preds > 0.5).long() == y.long()).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_acc / n


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.float().to(device)
            preds = model(x).squeeze()
            loss = criterion(preds, y)
            total_loss += loss.item() * x.size(0)
            total_acc += ((preds > 0.5).long() == y.long()).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_acc / n


def run_experiment(rnn_type="RNN", hidden_dim=32, max_len=200, subset=100000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 10000
    embed_dim = 64

    x_train, y_train, x_test, y_test = load_imdb(vocab_size)

    # x_train = x_train[:subset]
    # y_train = y_train[:subset]

    train_ds = list(zip(x_train, y_train))
    test_ds = list(zip(x_test, y_test))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              collate_fn=lambda b: pad_batch(b, max_len))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             collate_fn=lambda b: pad_batch(b, max_len))

    model = SentimentRNN(vocab_size, embed_dim, hidden_dim, rnn_type).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    test_losses = []

    for epoch in range(10):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = eval_model(model, test_loader, criterion, device)
        print(f"[{rnn_type}] h={hidden_dim} maxlen={max_len} | Epoch {epoch+1} | "
              f"Train acc={tr_acc:.3f} Test acc={te_acc:.3f}")
        train_losses.append(tr_loss)
        test_losses.append(te_loss)

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, marker='o', label='Train loss')
    plt.plot(epochs, test_losses, marker='o', label='Test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch | {rnn_type}, hidden={hidden_dim}, max_len={max_len}")
    plt.legend()

    fname = f"loss_{rnn_type}_h{hidden_dim}_max{max_len}.png"
    plt.savefig(fname, dpi=150)
    print(f"Saved loss plot to: {fname}")

    return tr_acc, te_acc

# Example runs
if __name__ == "__main__":
    for rnn_type in ["LSTM", "RNN"]:
        run_experiment(rnn_type=rnn_type, hidden_dim=64, max_len=100)
    for hidden_dim in [32, 64, 128]:
        run_experiment(rnn_type="LSTM", hidden_dim=hidden_dim, max_len=100)
    for max_len in [10, 20, 100]:
        run_experiment(rnn_type="LSTM", hidden_dim=64, max_len=max_len)


