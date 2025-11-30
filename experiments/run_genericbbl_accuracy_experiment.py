import sys
import os
import json

# Add project root to path to allow relative imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.attacks.membership_inference import lira_attack
from src.attacks.canary_mi import lira_canary_attack, label_only_canary_attack
from src.genericbbl.predict_genericbbl import train_genericbbl
from src.datasets.load_cifar10 import load_torch_dataset

def run_epsilon_accuracy_experiment():
    """
    Runs the GenericBBL training and evaluation for a range of epsilon values
    to measure the trade-off between privacy and utility (accuracy).
    """
    # Epsilon values to test, from more private to less private
    epsilon_values = [75, 500, 4000, 1e8, 1e9]
    results = []

    print("--- Loading CIFAR-10 binary dataset ---")
    try:
        # The `load_torch_dataset` function is assumed to be in the path
        train_data, test_data = load_torch_dataset("cifar10_binary")
    except FileNotFoundError:
        print("Error: Binary CIFAR-10 dataset not found in 'cifar10_binary/'.")
        print("Please run `src/datasets/load_cifar10.py` to generate it first.")
        return

    # Use a subset of the data for faster experiments, consistent with tests
    train_size = 10000
    test_size = 2000
    small_train_data = {
        "images": train_data["images"][:train_size],
        "labels": train_data["labels"][:train_size]
    }
    small_test_data = {
        "images": test_data["images"][:test_size],
        "labels": test_data["labels"][:test_size]
    }

    save_dir = os.path.join(ROOT, "results", "metrics")
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over the specified epsilon values
    for epsilon in epsilon_values:
        print(f"\n{'='*20} Running Experiment for Epsilon = {epsilon} {'='*20}")
        
        # Each call to train_genericbbl will train and evaluate the model
        # for the given epsilon. We use parameters that proved effective in tests.
        _, accuracy, final_epsilon = train_genericbbl(
            train_data=small_train_data,
            test_data=small_test_data,
            epochs=10,
            batch_size=128,
            epsilon=float(epsilon),  # Ensure epsilon is a float
            target_delta=1e-5,
            save_dir=save_dir,
            eval=True
        )

        if accuracy is not None:
            print(f"--- Epsilon = {epsilon}, Final Accuracy = {accuracy:.4f} ---")
            results.append({
                "epsilon": final_epsilon,
                "accuracy": accuracy
            })

    # Save the consolidated results to a new JSON file
    summary_filename = "genericbbl_epsilon_vs_accuracy.json"
    summary_path = os.path.join(save_dir, summary_filename)
    
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nExperiment run complete. All results saved to {summary_path}")
    print("This file can now be used to plot the privacy-utility trade-off.")

if __name__ == "__main__":
    run_epsilon_accuracy_experiment()
