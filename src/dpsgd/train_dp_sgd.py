import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import time, json, os

from src.datasets.load_adult import load_adult
from src.models.adult_mlp import AdultMLP

def train_dp_sgd(
    epochs=10,
    lr=1e-3,
    batch_size=256,
    max_grad_norm=1.0,
    noise_multiplier=1.1,
    target_delta=1e-5,
    save_dir="./results/metrics"
):
    train_loader, test_loader, input_dim = load_adult(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdultMLP(input_dim).to(device)
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

    start = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        eps = privacy_engine.get_epsilon(delta=target_delta)
        print(f"Epoch {epoch+1}/{epochs}: Loss={running_loss/len(train_loader):.3f}, ε={eps:.2f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).view(-1, 1)
            preds = (model(X) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    accuracy = correct / total

    metrics = {
        "dataset": "UCI Adult",
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "max_grad_norm": max_grad_norm,
        "noise_multiplier": noise_multiplier,
        "delta": target_delta,
        "epsilon": eps,
        "accuracy": accuracy,
        "runtime_sec": time.time() - start
    }

    os.makedirs(save_dir, exist_ok=True)
    json.dump(metrics, open(os.path.join(save_dir, "dpsgd_adult.json"), "w"), indent=4)
    print(f"Saved metrics to {save_dir}/dpsgd_adult.json")

if __name__ == "__main__":
    train_dp_sgd()
