import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def get_dataloaders(batch_size=32, train_percent=1.0):

    transform = transforms.ToTensor()

    train_data = torchvision.datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform
    )

    if train_percent < 1.0:
        subset_size = int(len(train_data) * train_percent)
        indices = torch.randperm(len(train_data))[:subset_size]
        train_data = Subset(train_data, indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class OneLayerNet(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TwoLayerNet(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def add_noise(inputs, noise_std=0.0):
    noise = torch.randn_like(inputs) * noise_std
    return inputs + noise


def train(model, train_loader, test_loader, noise_train=0.0, noise_test=0.0, epochs=5, lr=0.001):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    accuracies = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = add_noise(inputs, noise_train)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_losses.append(np.mean(batch_losses))


        model.eval()
        test_batch_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = add_noise(inputs, noise_test)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_batch_losses.append(loss.item())

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_losses.append(np.mean(test_batch_losses))
        accuracy = correct / total
        accuracies.append(accuracy)

        print(f"Epoch {epoch+1}: train_loss={train_losses[-1]:.4f}, "
              f"test_loss={test_losses[-1]:.4f}, acc={accuracy:.4f}")

    return train_losses, test_losses, accuracies



model_classes = {
    "OneLayer": OneLayerNet,
    "TwoLayer": TwoLayerNet,
}

batch_sizes = [
    # 16,
    # 32,
    64
]
train_percents = [
    1.0,
    # 0.1,
    # 0.01
]
hidden_sizes = [
    # 64,
    128,
    # 256
]
noise_settings = [
    (0.0, 0.0),
    (0.0, 0.3),
    (0.3, 0.3),
]

import csv

results_file = "results_noise_train1.0_bs64_h128.csv"
if os.path.exists(results_file):
    os.remove(results_file)

with open(results_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "experiment_id",
        "model",
        "batch_size",
        "train_percent",
        "hidden_size",
        "noise_train",
        "noise_test",
        "final_train_loss",
        "final_test_loss",
        "accuracy",
        "plot_file"
    ])


def log(msg):
    print(msg)


experiment_id = 0

for model_name, model_class in model_classes.items():
    for batch_size in batch_sizes:
        for train_percent in train_percents:
            for hidden_size in hidden_sizes:
                for noise_train, noise_test in noise_settings:

                    experiment_id += 1
                    log("=" * 80)
                    log(f"Experiment {experiment_id}")
                    log(f"Model: {model_name}")
                    log(f"Batch size: {batch_size}")
                    log(f"Train percent: {train_percent}")
                    log(f"Hidden size: {hidden_size}")
                    log(f"Noise (train, test): ({noise_train}, {noise_test})")
                    log("-" * 80)


                    train_loader, test_loader = get_dataloaders(
                        batch_size=batch_size,
                        train_percent=train_percent
                    )

                    model = model_class(hidden_size=hidden_size)


                    train_losses, test_losses, accuracies = train(
                        model,
                        train_loader,
                        test_loader,
                        noise_train=noise_train,
                        noise_test=noise_test,
                        epochs=20,
                        lr=0.001,
                    )

                    final_train = train_losses[-1]
                    final_test = test_losses[-1]
                    final_accuracy = accuracies[-1]

                    log(f"Final train loss: {final_train:.4f}")
                    log(f"Final test loss:  {final_test:.4f}")


                    plot_name = (
                        f"plot_"
                        f"{model_name}_"
                        f"bs{batch_size}_"
                        f"train{train_percent}_"
                        f"h{hidden_size}_"
                        f"noise{noise_train}-{noise_test}.png"
                    )

                    plt.figure()
                    plt.plot(train_losses, label="Train loss")
                    plt.plot(test_losses, label="Test loss")
                    plt.legend()
                    plt.title(
                        f"{model_name}, bs={batch_size}, train%={train_percent}, "
                        f"h={hidden_size}, noise=({noise_train},{noise_test})"
                    )
                    plt.savefig(plot_name)
                    plt.close()

                    log(f"Plot saved: {plot_name}\n")

                    with open(results_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            experiment_id,
                            model_name,
                            batch_size,
                            train_percent,
                            hidden_size,
                            noise_train,
                            noise_test,
                            round(final_train, 6),
                            round(final_test, 6),
                            round(final_accuracy, 6),
                            plot_name
                        ])

log("ALL EXPERIMENTS FINISHED.")
