import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.baseline.train_baseline import train_baseline_cifar10

if __name__ == "__main__":
    train_baseline_cifar10(
        epochs=10,
        lr=1e-3,
        batch_size=128,
        save_dir="./results/metrics"
    )
