import sys, os
import json
from pathlib import Path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.attacks.membership_inference import lira_attack, label_only_attack
from src.attacks.canary_mi import lira_canary_attack, label_only_canary_attack
from src.dpsgd.train_dp_sgd import train_dp_sgd
from src.baseline.train_baseline import train_baseline_cifar10
from src.genericbbl.predict_genericbbl import train_genericbbl


def get_wrapper_training_func(training_func, **kwargs):
    """Create a wrapper function with fixed hyperparameters."""
    def wrapper(train_data, test_data, eval=True):
        return training_func(train_data, test_data, **kwargs, eval=eval)
    wrapper.__name__ = training_func.__name__
    return wrapper


def save_attack_result(result, filename_prefix, num_models, target_fpr):
    """
    Save attack result with unique filename.
    
    Args:
        result: Attack result dictionary
        filename_prefix: Prefix for filename (includes algorithm and attack type)
        num_models: Number of models used
        target_fpr: Target FPR
    """
    output_dir = Path("evaluation/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{filename_prefix}_{num_models}models_fpr{target_fpr:.2f}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Saved to: {output_file}")


def run_all_attacks(hyperparams_file="experiments/hyperparams.json", num_models=100, target_fpr=0.01):
    """
    Run all attacks on all algorithm configurations from hyperparams.
    
    Args:
        hyperparams_file: Path to hyperparameters JSON file
        num_models: Number of models to train for attacks
        target_fpr: Target false positive rate
    """
    print("=" * 80)
    print("Running Membership Inference Attacks on All Configurations")
    print("=" * 80)
    
    # Load hyperparameters
    with open(hyperparams_file, 'r') as f:
        hyperparams = json.load(f)
    
    all_results = []
    
    # # 1. Baseline attacks
    # print("\n" + "=" * 80)
    # print("BASELINE MODEL ATTACKS")
    # print("=" * 80)
    
    # baseline_params = hyperparams.get("train_baseline", [])
    # if baseline_params:
    #     params = baseline_params[0]  # Baseline has only one config
    #     print(f"\nBaseline parameters: {params}")
        
    #     wrapper = get_wrapper_training_func(
    #         train_baseline_cifar10,
    #         epochs=params.get("epochs", 20),
    #         lr=params.get("lr", 1e-3),
    #         batch_size=params.get("batch_size", 128)
    #     )
        
    #     # LIRA attack
    #     print("\n" + "-" * 80)
    #     print("Running LIRA Canary Attack on Baseline...")
    #     print("-" * 80)
    #     tpr, results = lira_canary_attack(num_models, wrapper, target_fpr=target_fpr)
    #     results["algorithm"] = "baseline"
    #     results["hyperparameters"] = params
    #     all_results.append(results)
        
    #     # Save individual result
    #     save_attack_result(results, "baseline_lira", num_models, target_fpr)
        
    #     # Label-only attack
    #     print("\n" + "-" * 80)
    #     print("Running Label-Only Canary Attack on Baseline...")
    #     print("-" * 80)
    #     tpr, results = label_only_canary_attack(num_models, wrapper, target_fpr=target_fpr)
    #     results["algorithm"] = "baseline"
    #     results["hyperparameters"] = params
    #     all_results.append(results)
        
    #     # Save individual result
    #     save_attack_result(results, "baseline_label_only", num_models, target_fpr)
    
    # # 2. DP-SGD attacks
    # print("\n" + "=" * 80)
    # print("DP-SGD MODEL ATTACKS")
    # print("=" * 80)
    
    # dpsgd_params = hyperparams.get("train_dp_sgd", [])
    # for idx, params in enumerate(dpsgd_params, 1):
    #     print(f"\n{'=' * 80}")
    #     print(f"DP-SGD Configuration {idx}/{len(dpsgd_params)}")
    #     print(f"Parameters: {params}")
    #     print("=" * 80)
        
    #     wrapper = get_wrapper_training_func(
    #         train_dp_sgd,
    #         epochs=params.get("epochs", 20),
    #         lr=params.get("lr", 1e-3),
    #         batch_size=params.get("batch_size", 128),
    #         max_grad_norm=params.get("max_grad_norm", 1.0),
    #         noise_multiplier=params.get("noise_multiplier", 1.0),
    #         target_delta=params.get("target_delta", 1e-5)
    #     )
        
    #     # LIRA attack
    #     print("\n" + "-" * 80)
    #     print(f"Running LIRA Canary Attack on DP-SGD (noise={params.get('noise_multiplier')})...")
    #     print("-" * 80)
    #     noise = params.get('noise_multiplier')
    #     tpr, results = lira_canary_attack(num_models, wrapper, target_fpr=target_fpr, 
    #                                      output_prefix=f"canary_lira_dpsgd_nm{noise}")
    #     results["algorithm"] = "dpsgd"
    #     results["hyperparameters"] = params
    #     all_results.append(results)
        
    #     # Save with unique filename
    #     save_attack_result(results, f"dpsgd_nm{noise}_lira", num_models, target_fpr)
        
    #     # Label-only attack
    #     print("\n" + "-" * 80)
    #     print(f"Running Label-Only Canary Attack on DP-SGD (noise={noise})...")
    #     print("-" * 80)
    #     tpr, results = label_only_canary_attack(num_models, wrapper, target_fpr=target_fpr,
    #                                            output_prefix=f"canary_label_only_dpsgd_nm{noise}")
    #     results["algorithm"] = "dpsgd"
    #     results["hyperparameters"] = params
    #     all_results.append(results)
        
    #     # Save with unique filename
    #     save_attack_result(results, f"dpsgd_nm{noise}_label_only", num_models, target_fpr)
    
    # 3. GenericBBL attacks (Label-only only)
    print("\n" + "=" * 80)
    print("GENERICBBL MODEL ATTACKS")
    print("=" * 80)
    
    genericbbl_params = hyperparams.get("train_genericbbl", [])
    for idx, params in enumerate(genericbbl_params, 1):
        print(f"\n{'=' * 80}")
        print(f"GenericBBL Configuration {idx}/{len(genericbbl_params)}")
        print(f"Parameters: {params}")
        print("=" * 80)
        
        wrapper = get_wrapper_training_func(
            train_genericbbl,
            epochs=params.get("epochs", 20),
            lr=params.get("lr", 1e-3),
            batch_size=params.get("batch_size", 128),
            epsilon=params.get("epsilon", 100),
        )
        
        # Label-only attack (GenericBBL only supports label-only)
        print("\n" + "-" * 80)
        epsilon = params.get('epsilon')
        print(f"Running Label-Only Canary Attack on GenericBBL (epsilon={epsilon})...")
        print("-" * 80)
        tpr, results = label_only_canary_attack(num_models, wrapper, target_fpr=target_fpr,
                                               output_prefix=f"canary_label_only_genericbbl_eps{epsilon}")
        results["algorithm"] = "genericbbl"
        results["hyperparameters"] = params
        all_results.append(results)
        
        # Save with unique filename
        save_attack_result(results, f"genericbbl_eps{epsilon}_label_only", num_models, target_fpr)
    
    # Save summary of all results
    summary_file = "evaluation/data/all_attacks_summary.json"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ALL ATTACKS COMPLETED!")
    print("=" * 80)
    print(f"\nSummary saved to: {summary_file}")
    print(f"\nTotal experiments run: {len(all_results)}")
    
    # Print summary table
    print("\nSummary of Attack Results:")
    print(f"{'Algorithm':<15} {'Attack Type':<15} {'Noise/Eps':<12} {'TPR @ FPR={target_fpr}':<20}")
    print("-" * 65)
    for result in all_results:
        algo = result["algorithm"]
        attack = result["attack_type"]
        tpr = result["tpr"]
        
        if algo == "dpsgd":
            param_val = f"σ={result['hyperparameters']['noise_multiplier']}"
        elif algo == "genericbbl":
            param_val = f"ε={result['hyperparameters']['epsilon']}"
        else:
            param_val = "N/A"
        
        print(f"{algo:<15} {attack:<15} {param_val:<12} {tpr:<20.4f}")
    
    return all_results


if __name__ == "__main__":
    run_all_attacks(
        hyperparams_file="experiments/hyperparams.json",
        num_models=100,
        target_fpr=0.01
    )
