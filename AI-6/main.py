import argparse
import itertools
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class AddGaussianNoise(object):
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, tensor):
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise

    def __repr__(self):
        return f"AddGaussianNoise(std={self.std})"


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, conv_out_channels=16, kernel_size=3, pool_size=2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, conv_out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(pool_size)

        self.flatten = nn.Flatten()
        self.classifier = nn.LazyLinear(10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def get_dataloaders(batch_size, noise_std_train=0.0, noise_std_test=0.0):

    base_transform = [transforms.ToTensor()]

    train_transform = list(base_transform)
    if noise_std_train > 0:
        train_transform.append(AddGaussianNoise(noise_std_train))
    train_transform = transforms.Compose(train_transform)

    test_transform = list(base_transform)
    if noise_std_test > 0:
        test_transform.append(AddGaussianNoise(noise_std_test))
    test_transform = transforms.Compose(test_transform)

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


def train_epoch(model, device, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


def eval_model(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return running_loss / total, correct / total


def run_experiment(args, conv_out_channels, kernel_size, pool_size, noise_std, noise_mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Noise
    if noise_mode == 'none':
        n_train = 0.0
        n_test = 0.0
    elif noise_mode == 'test_only':
        n_train = 0.0
        n_test = noise_std
    elif noise_mode == 'train_and_test':
        n_train = noise_std
        n_test = noise_std
    else:
        raise ValueError("Unknown noise_mode")

    train_loader, test_loader = get_dataloaders(args.batch_size, noise_std_train=n_train, noise_std_test=n_test)

    model = SimpleCNN(in_channels=1, conv_out_channels=conv_out_channels, kernel_size=kernel_size, pool_size=pool_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_test_acc = 0.0
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = eval_model(model, device, test_loader, criterion)
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if args.verbose:
            print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | test loss {test_loss:.4f} acc {test_acc:.4f}")

    return best_test_acc, history


def main():

    class Args:
        epochs = 5
        batch_size = 128
        lr = 1e-3
        out_channels = [16]
        kernel_sizes = [3]
        pool_sizes = [2]
        noise_levels = [0.3]
        noise_mode = ['none', 'test_only', 'train_and_test']
        # out_channels = [8, 16, 32]
        # kernel_sizes = [3, 5]
        # pool_sizes = [2, 3]
        # noise_levels = [0.0, 0.3]
        # noise_mode = ['none', 'test_only', 'train_and_test']
        results_dir = 'results'
        verbose = True
    args = Args()

    os.makedirs(args.results_dir, exist_ok=True)

    combinations = list(itertools.product(args.out_channels, args.kernel_sizes, args.pool_sizes, args.noise_levels, args.noise_mode))
    print(f"Running {len(combinations)} experiments (out_channels x kernel x pool x noise)")

    records = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for (c_out, k, p, n, m) in combinations:
        print(f"=== Experiment: channels={c_out}, kernel={k}, pool={p}, noise={n}, mode={m} ===")
        best_acc, history = run_experiment(args, conv_out_channels=c_out, kernel_size=k, pool_size=p, noise_std=n, noise_mode=m)
        rec = {
            'channels': c_out,
            'kernel_size': k,
            'pool_size': p,
            'noise_std': n,
            'noise_mode': m,
            'best_test_acc': best_acc
        }
        records.append(rec)
        # save per-experiment history
        hist_df = pd.DataFrame(history)
        hist_fname = os.path.join(args.results_dir, f"hist_ch{c_out}_k{k}_p{p}_n{n}_m{m}_{timestamp}.csv")
        hist_df.to_csv(hist_fname, index=False)

    results_df = pd.DataFrame(records)
    results_fname = os.path.join(args.results_dir, f"summary_{args.noise_mode}_{timestamp}.csv")
    results_df.to_csv(results_fname, index=False)
    print(f"All experiments finished. Summary saved to {results_fname}")

    for (c_out, k, p, n, m) in combinations:
        hist_file = os.path.join(args.results_dir, f"hist_ch{c_out}_k{k}_p{p}_n{n}_m{m}_{timestamp}.csv")
        if not os.path.exists(hist_file):
            continue
        hist_df = pd.read_csv(hist_file)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hist_df['epoch'], hist_df['train_loss'], marker='o', label='train loss')
        ax.plot(hist_df['epoch'], hist_df['test_loss'], marker='o', label='test loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title(f'Loss vs epoch | ch={c_out}, k={k}, p={p}, noise={n}')
        ax.legend()

        plot_fname = os.path.join(args.results_dir, f"loss_ch{c_out}_k{k}_p{p}_n{n}_m{m}_{timestamp}.png")
        fig.savefig(plot_fname, dpi=150)
        print(f"Saved loss plot: {plot_fname}")


if __name__ == '__main__':
    # print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())
    # a = torch.randn((4096, 4096), device="cuda")
    # b = torch.randn((4096, 4096), device="cuda")
    # c = a @ b
    # print("Success!", c.shape)
    main()
