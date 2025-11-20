import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dpsgd.train_dp_sgd import train_dp_sgd
from src.datasets.load_cifar10 import load_torch_dataset

if __name__ == "__main__":
    # Load data
    train_data, test_data = load_torch_dataset("cifar10_binary")
    
    # Train model with DP-SGD
    train_dp_sgd(
        train_data,
        test_data,
        epochs=10,
        lr=1e-3,
        batch_size=256,
        max_grad_norm=1.0,
        noise_multiplier=1.1,
        target_delta=1e-5,
        save_dir="./results/metrics"
    )
