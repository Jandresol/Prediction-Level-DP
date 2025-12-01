import sys, os
import json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dpsgd.train_dp_sgd import train_dp_sgd
from src.datasets.load_cifar10 import load_torch_dataset


def run_dpsgd_experiments(hyperparams_file="experiments/hyperparams.json"):
    """
    Run DP-SGD experiments with different hyperparameters.
    
    Args:
        hyperparams_file: Path to hyperparameters JSON file
    """
    # Load hyperparameters
    print("=" * 80)
    print("Running DP-SGD Experiments with Multiple Hyperparameters")
    print("=" * 80)
    
    with open(hyperparams_file, 'r') as f:
        hyperparams = json.load(f)
    
    dpsgd_params = hyperparams.get("train_dp_sgd", [])
    
    if not dpsgd_params:
        print("No DP-SGD hyperparameters found in the file!")
        return
    
    print(f"\nFound {len(dpsgd_params)} hyperparameter configurations")
    
    # Load data once
    print("\nLoading CIFAR-10 binary dataset...")
    train_data, test_data = load_torch_dataset("cifar10_binary")
    print(f"Training samples: {len(train_data['images'])}")
    print(f"Test samples: {len(test_data['images'])}")
    
    # Store results
    results = []
    
    # Run experiments
    for idx, params in enumerate(dpsgd_params, 1):
        print("\n" + "=" * 80)
        print(f"Experiment {idx}/{len(dpsgd_params)}")
        print("=" * 80)
        print(f"Parameters: {params}")
        
        # Set default values
        epochs = params.get("epochs", 20)
        batch_size = params.get("batch_size", 128)
        noise_multiplier = params.get("noise_multiplier", 1.0)
        lr = params.get("lr", 1e-3)
        max_grad_norm = params.get("max_grad_norm", 1.0)
        target_delta = params.get("target_delta", 1e-5)
        
        # Train model
        model, accuracy, epsilon = train_dp_sgd(
            train_data,
            test_data,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            save_dir="./results/dpsgd_sweeps",
            eval=True
        )
        
        # Store results
        result = {
            "experiment_id": idx,
            "hyperparameters": params,
            "epsilon": epsilon,
            "accuracy": accuracy,
            "delta": target_delta
        }
        results.append(result)
        
        print(f"\n{'=' * 40}")
        print(f"Results for Experiment {idx}:")
        print(f"  Epsilon (ε): {epsilon:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Noise Multiplier: {noise_multiplier}")
        print(f"{'=' * 40}")
    
    # Save all results
    output_file = "results/dpsgd_sweeps/dpsgd_epsilon_accuracy_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("All Experiments Completed!")
    print("=" * 80)
    
    # Print summary table
    print("\nSummary of Results:")
    print(f"{'Exp':<5} {'Noise Mult.':<15} {'Epsilon (ε)':<15} {'Accuracy':<15}")
    print("-" * 50)
    for result in results:
        exp_id = result["experiment_id"]
        noise = result["hyperparameters"]["noise_multiplier"]
        eps = result["epsilon"]
        acc = result["accuracy"]
        print(f"{exp_id:<5} {noise:<15.2f} {eps:<15.4f} {acc:<15.4f}")
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    run_dpsgd_experiments()
