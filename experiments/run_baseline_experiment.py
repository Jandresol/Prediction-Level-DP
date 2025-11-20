import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.baseline.train_baseline import train_baseline_cifar10
from src.datasets.load_cifar10 import load_torch_dataset

if __name__ == "__main__":
    # Load data
    train_data, test_data = load_torch_dataset("cifar10_binary")
    
    # Train model
    train_baseline_cifar10(
        train_data,
        test_data,
        epochs=10,
        lr=1e-3,
        batch_size=128,
        save_dir="./results/metrics"
    )
