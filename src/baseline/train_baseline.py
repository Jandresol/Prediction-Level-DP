import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time, json, os

from src.datasets.load_cifar10 import load_torch_dataset
from src.models.cifar_cnn import cifar_cnn


def train_baseline_cifar10(
    train_data,
    test_data,
    epochs=10,
    lr=1e-3,
    batch_size=128,
    save_dir="./results/metrics"
):

    # --------------------------
    # Process data
    # --------------------------
    X_train = train_data["images"].float() / 255.0
    y_train = train_data["labels"].float()

    X_test = test_data["images"].float() / 255.0
    y_test = test_data["labels"].float()

    # Wrap datasets
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    # --------------------------
    # Model
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cifar_cnn().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # --------------------------
    # Training
    # --------------------------
    start = time.time()

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device).view(-1, 1)

            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss={loss_sum/len(train_loader):.4f}")

    # --------------------------
    # Evaluation
    # --------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).view(-1, 1)
            preds = (model(X) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    runtime = time.time() - start

    # --------------------------
    # Save metrics
    # --------------------------
    metrics = {
        "dataset": "CIFAR-10 binary",
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "accuracy": accuracy,
        "runtime_sec": runtime,
    }

    os.makedirs(save_dir, exist_ok=True)
    json.dump(
        metrics,
        open(os.path.join(save_dir, "baseline_cifar10.json"), "w"),
        indent=4,
    )

    print(f"Saved baseline metrics to {save_dir}/baseline_cifar10.json")


if __name__ == "__main__":
    train_data, test_data = load_torch_dataset("cifar10_binary")
    train_baseline_cifar10(train_data, test_data)
