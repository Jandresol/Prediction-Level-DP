import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import time, json, os

from src.datasets.load_cifar10 import load_torch_dataset
from src.models.cifar_cnn import cifar_cnn


def train_dp_sgd(
    epochs=10,
    lr=1e-3,
    batch_size=128,
    max_grad_norm=1.0,
    noise_multiplier=1.1,
    target_delta=1e-5,
    save_dir="./results/metrics"
):
    # --------------------------
    # Load the tensors (dicts)
    # --------------------------
    train_data, test_data = load_torch_dataset("cifar10_binary")

    X_train = train_data["images"].float() / 255.0
    y_train = train_data["labels"].float()

    X_test = test_data["images"].float() / 255.0
    y_test = test_data["labels"].float()

    # --------------------------
    # Wrap into DataLoaders
    # --------------------------
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --------------------------
    # Model + DP setup
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cifar_cnn().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

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

        eps = privacy_engine.get_epsilon(delta=target_delta)
        print(f"Epoch {epoch+1}/{epochs}  Loss={loss_sum/len(train_loader):.4f}  ε={eps:.2f}")

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

    # --------------------------
    # Save metrics
    # --------------------------
    metrics = {
        "dataset": "CIFAR-10 binary",
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "max_grad_norm": max_grad_norm,
        "noise_multiplier": noise_multiplier,
        "delta": target_delta,
        "epsilon": eps,
        "accuracy": accuracy,
        "runtime_sec": time.time() - start,
    }

    os.makedirs(save_dir, exist_ok=True)
    json.dump(metrics, open(os.path.join(save_dir, "dpsgd_cifar10.json"), "w"), indent=4)
    print(f"Saved metrics to {save_dir}/dpsgd_cifar10.json")


if __name__ == "__main__":
    train_dp_sgd()