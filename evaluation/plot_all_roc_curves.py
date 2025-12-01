"""
Plot All ROC Curves from Attack Results

This script reads all attack results from evaluation/data/ and plots
their ROC curves together for comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_all_results(data_dir="evaluation/data"):
    """
    Load all attack results from the data directory.
    
    Args:
        data_dir: Directory containing attack result JSON files
        
    Returns:
        List of result dictionaries
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Data directory {data_dir} does not exist!")
        return []
    
    # Load all JSON files
    results = []
    json_files = list(data_path.glob("*.json"))
    
    print(f"Found {len(json_files)} result files")
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            result = json.load(f)
            result["filename"] = json_file.name
            results.append(result)
    
    return results


def create_combined_roc_plot(results, output_file="evaluation/figures/combined_roc_curves.png"):
    """
    Create a combined ROC curve plot for all attack results.
    
    Args:
        results: List of attack result dictionaries
        output_file: Path to save the plot
    """
    if not results:
        print("No results to plot!")
        return
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color schemes for different algorithms
    colors = {
        "baseline": "blue",
        "dpsgd": "red",
        "genericbbl": "green"
    }
    
    # Line styles for attack types
    linestyles = {
        "LIRA": "-",
        "Label-Only": "--"
    }
    
    # Plot each ROC curve
    for result in results:
        algorithm = result.get("algorithm", "unknown")
        attack_type = result.get("attack_type", "unknown")
        hyperparams = result.get("hyperparameters", {})
        
        # Get ROC curve data
        if "roc_curve" not in result:
            print(f"Skipping {result.get('filename')}: no ROC curve data")
            continue
        
        fprs = np.array(result["roc_curve"]["fprs"])
        tprs = np.array(result["roc_curve"]["tprs"])
        
        # Ensure positive values for log scale
        fprs = np.maximum(fprs, 1e-6)
        tprs = np.maximum(tprs, 1e-6)
        
        # Create label
        if algorithm == "baseline":
            label = f"Baseline - {attack_type}"
        elif algorithm == "dpsgd":
            noise = hyperparams.get("noise_multiplier", "?")
            label = f"DP-SGD (σ={noise}) - {attack_type}"
        elif algorithm == "genericbbl":
            epsilon = hyperparams.get("epsilon", "?")
            label = f"GenericBBL (ε={epsilon}) - {attack_type}"
        else:
            label = f"{algorithm} - {attack_type}"
        
        # Plot
        color = colors.get(algorithm, "gray")
        linestyle = linestyles.get(attack_type, "-")
        
        ax.plot(fprs, tprs, 
               color=color,
               linestyle=linestyle,
               linewidth=2,
               label=label,
               alpha=0.8)
    
    # Random guess line
    log_fprs = np.logspace(-6, 0, 100)
    ax.plot(log_fprs, log_fprs, 'k--', alpha=0.3, linewidth=1.5, label='Random Guess')
    
    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves: Membership Inference Attacks on Different Algorithms', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-3, 1])
    ax.set_ylim([1e-3, 1])
    
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Combined ROC plot saved to: {output_file}")
    plt.close()


def create_algorithm_comparison_plots(results, output_dir="evaluation/figures"):
    """
    Create separate ROC plots comparing different noise levels for each algorithm.
    
    Args:
        results: List of attack result dictionaries
        output_dir: Directory to save the plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group by algorithm
    by_algorithm = {}
    for result in results:
        algo = result.get("algorithm", "unknown")
        if algo not in by_algorithm:
            by_algorithm[algo] = []
        by_algorithm[algo].append(result)
    
    # Plot for each algorithm
    for algo, algo_results in by_algorithm.items():
        if not algo_results:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each configuration
        for result in algo_results:
            attack_type = result.get("attack_type", "unknown")
            hyperparams = result.get("hyperparameters", {})
            
            if "roc_curve" not in result:
                continue
            
            fprs = np.array(result["roc_curve"]["fprs"])
            tprs = np.array(result["roc_curve"]["tprs"])
            
            fprs = np.maximum(fprs, 1e-6)
            tprs = np.maximum(tprs, 1e-6)
            
            # Create label based on algorithm
            if algo == "dpsgd":
                noise = hyperparams.get("noise_multiplier", "?")
                label = f"σ={noise} - {attack_type}"
            elif algo == "genericbbl":
                epsilon = hyperparams.get("epsilon", "?")
                label = f"ε={epsilon} - {attack_type}"
            else:
                label = attack_type
            
            linestyle = "--" if attack_type == "Label-Only" else "-"
            ax.plot(fprs, tprs, linewidth=2, label=label, linestyle=linestyle, alpha=0.8)
        
        # Random guess
        log_fprs = np.logspace(-6, 0, 100)
        ax.plot(log_fprs, log_fprs, 'k--', alpha=0.3, linewidth=1.5, label='Random Guess')
        
        # Formatting
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'ROC Curves: {algo.upper()} with Different Configurations', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1e-3, 1])
        ax.set_ylim([1e-3, 1])
        
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        output_file = output_path / f"roc_curves_{algo}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ {algo.upper()} ROC plot saved to: {output_file}")
        plt.close()


def create_attack_type_comparison(results, output_dir="evaluation/figures"):
    """
    Create plots comparing LIRA vs Label-Only attacks.
    
    Args:
        results: List of attack result dictionaries
        output_dir: Directory to save the plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Group by attack type
    attack_groups = {"LIRA": [], "Label-Only": []}
    
    for result in results:
        attack_type = result.get("attack_type", "unknown")
        if attack_type in attack_groups and "roc_curve" in result:
            attack_groups[attack_type].append(result)
    
    # Plot each attack type
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for attack_type, attack_results in attack_groups.items():
        for idx, result in enumerate(attack_results):
            algo = result.get("algorithm", "unknown")
            hyperparams = result.get("hyperparameters", {})
            
            fprs = np.array(result["roc_curve"]["fprs"])
            tprs = np.array(result["roc_curve"]["tprs"])
            
            fprs = np.maximum(fprs, 1e-6)
            tprs = np.maximum(tprs, 1e-6)
            
            # Create label
            if algo == "dpsgd":
                noise = hyperparams.get("noise_multiplier", "?")
                label = f"{attack_type}: DP-SGD (σ={noise})"
            elif algo == "genericbbl":
                epsilon = hyperparams.get("epsilon", "?")
                label = f"{attack_type}: GenericBBL (ε={epsilon})"
            else:
                label = f"{attack_type}: {algo}"
            
            linestyle = "--" if attack_type == "Label-Only" else "-"
            ax.plot(fprs, tprs, linewidth=2, label=label, 
                   linestyle=linestyle, alpha=0.7)
    
    # Random guess
    log_fprs = np.logspace(-6, 0, 100)
    ax.plot(log_fprs, log_fprs, 'k--', alpha=0.3, linewidth=1.5, label='Random Guess')
    
    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves: Comparing Attack Types', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-3, 1])
    ax.set_ylim([1e-3, 1])
    
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_file = output_path / "roc_curves_by_attack_type.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Attack type comparison plot saved to: {output_file}")
    plt.close()


def main():
    """Main function to generate all ROC curve plots."""
    print("=" * 80)
    print("Generating ROC Curve Plots from Attack Results")
    print("=" * 80)
    
    # Load all results
    print("\nLoading results...")
    results = load_all_results("evaluation/data")
    
    if not results:
        print("No results found! Run attacks first.")
        return
    
    print(f"Loaded {len(results)} attack results")
    
    # Create plots
    print("\nGenerating plots...")
    print("-" * 80)
    
    # 1. Combined plot with all ROC curves
    create_combined_roc_plot(results)
    
    # 2. Per-algorithm comparison plots
    create_algorithm_comparison_plots(results)
    
    # 3. Attack type comparison
    create_attack_type_comparison(results)
    
    print("\n" + "=" * 80)
    print("All plots generated successfully!")
    print("=" * 80)
    print("\nPlots saved in: evaluation/figures/")


if __name__ == "__main__":
    main()

