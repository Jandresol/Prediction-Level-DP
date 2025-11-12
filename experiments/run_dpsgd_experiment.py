from src.dpsgd.train_dp_sgd import train_dp_sgd

if __name__ == "__main__":
    train_dp_sgd(
        epochs=10,
        lr=1e-3,
        batch_size=256,
        max_grad_norm=1.0,
        noise_multiplier=1.1,
        target_delta=1e-5,
        save_dir="./results/metrics"
    )
