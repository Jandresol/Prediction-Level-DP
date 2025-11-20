import torch
import torch.nn as nn
import torch.optim as optim
from src.datasets.load_adult import load_adult
from src.models.adult_mlp import AdultMLP
from tqdm import tqdm

def train_baseline(epochs=10, lr=1e-3, batch_size=128, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader, test_loader, input_dim, num_classes = load_adult(batch_size=batch_size)

    # Model
    model = AdultMLP(input_dim=input_dim, num_classes=num_classes).to(device)

    # Standard loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training baseline (non-DP) model on {device} for {epochs} epochs...")

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.3f}")

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            _, predicted = preds.max(1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    print(f"\nFinal Baseline Accuracy: {accuracy*100:.2f}%")
    return accuracy


if __name__ == "__main__":
    train_baseline()
