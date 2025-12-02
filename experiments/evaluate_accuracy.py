import sys
import os
import json
from pathlib import Path

# Add project root to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datasets.load_cifar10 import load_torch_dataset
from src.dpsgd.train_dp_sgd import train_dp_sgd
from src.genericbbl.predict_genericbbl import train_genericbbl


def evaluate_all_accuracies(hyperparams_file="experiments/hyperparams.json"):
    """
    Evaluates the accuracy of DPSGD and GenericBBL models based on configurations
    in the provided hyperparams file.
    """
    print("=" * 80)
    print("Running Accuracy Evaluation for DPSGD and GenericBBL")
    print("=" * 80)

    # Load hyperparameters
    with open(hyperparams_file, 'r') as f:
        hyperparams = json.load(f)

    # Load dataset
    print("\nLoading CIFAR-10 binary dataset...")
    try:
        train_data, test_data = load_torch_dataset("cifar10_binary")
        print("✓ Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: `cifar10_binary` dataset not found.")
        print("Please run `python src/datasets/load_cifar10.py` to generate it first.")
        return

    all_results = []

    # 1. Evaluate DP-SGD models
    print("\n" + "=" * 80)
    print("Evaluating DP-SGD Models")
    print("=" * 80)

    dpsgd_params_list = hyperparams.get("train_dp_sgd", [])
    for idx, params in enumerate(dpsgd_params_list, 1):
        print(f"\n--- DP-SGD Configuration {idx}/{len(dpsgd_params_list)} ---")
        print(f"Parameters: {params}")
        print("-" * 40)

        _, accuracy, final_epsilon = train_dp_sgd(
            train_data=train_data,
            test_data=test_data,
            epochs=params.get("epochs", 15),
            batch_size=params.get("batch_size", 128),
            noise_multiplier=params.get("noise_multiplier", 1.0),
            # Use defaults from train_dp_sgd if not in hyperparams
            lr=params.get("lr", 1e-3),
            max_grad_norm=params.get("max_grad_norm", 1.0),
            target_delta=params.get("target_delta", 1e-5),
            eval=True
        )

        result = {
            "algorithm": "dpsgd",
            "hyperparameters": params,
            "accuracy": accuracy,
            "final_epsilon": final_epsilon,
        }
        all_results.append(result)
        print(f"✓ Accuracy: {accuracy:.4f} at ε={final_epsilon:.2f}")

    # 2. Evaluate GenericBBL models
    print("\n" + "=" * 80)
    print("Evaluating GenericBBL Models")
    print("=" * 80)

    genericbbl_params_list = hyperparams.get("train_genericbbl", [])
    for idx, params in enumerate(genericbbl_params_list, 1):
        print(f"\n--- GenericBBL Configuration {idx}/{len(genericbbl_params_list)} ---")
        print(f"Parameters: {params}")
        print("-" * 40)

        _, accuracy, epsilon = train_genericbbl(
            train_data=train_data,
            test_data=test_data,
            epochs=params.get("epochs", 15),
            batch_size=params.get("batch_size", 128),
            epsilon=params.get("epsilon", 100),
            # Use defaults from train_genericbbl if not in hyperparams
            lr=params.get("lr", 1e-3),
            target_delta=params.get("target_delta", 1e-5),
            eval=True
        )

        result = {
            "algorithm": "genericbbl",
            "hyperparameters": params,
            "accuracy": accuracy,
            "final_epsilon": epsilon, # Target epsilon for BBL
        }
        all_results.append(result)
        print(f"✓ Accuracy: {accuracy:.4f} at ε={epsilon:.2f}")


    # Save summary of all results
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "accuracy_evaluation_summary.json"
    
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "=" * 80)
    print("ALL ACCURACY EVALUATIONS COMPLETED!")
    print("=" * 80)
    print(f"\nSummary of results saved to: {summary_file}")

    # Print summary table
    print("\n--- Summary of Accuracy Results ---")
    print(f"{'Algorithm':<15} {'Parameters':<25} {'Accuracy':<15} {'Epsilon (ε)':<15}")
    print("-" * 70)
    for result in all_results:
        algo = result["algorithm"]
        acc = result["accuracy"]
        eps = result["final_epsilon"]
        
        if algo == "dpsgd":
            param_str = f"noise={result['hyperparameters']['noise_multiplier']}"
        elif algo == "genericbbl":
            param_str = f"eps_target={result['hyperparameters']['epsilon']}"
        else:
            param_str = "N/A"
        
        print(f"{algo:<15} {param_str:<25} {acc:<15.4f} {eps:<15.2f}")
    print("-" * 70)

    return all_results


if __name__ == "__main__":
    evaluate_all_accuracies()
