"""
Plot Bar Charts of TPR at Specific FPR Thresholds

This script reads attack results and plots bar charts showing TPR values
at FPR = 0.01, 0.03, and 0.1 for all algorithm configurations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import interpolate


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
    
    # Load all JSON files (except summary)
    results = []
    for json_file in data_path.glob("*.json"):
        if "summary" in json_file.name:
            continue
            
        with open(json_file, 'r') as f:
            result = json.load(f)
            result["filename"] = json_file.name
            results.append(result)
    
    print(f"Found {len(results)} result files")
    return results


def get_tpr_at_fpr(fprs, tprs, target_fpr):
    """
    Interpolate TPR at a specific FPR value.
    
    Args:
        fprs: Array of FPR values
        tprs: Array of TPR values
        target_fpr: Target FPR to get TPR for
        
    Returns:
        TPR value at target FPR
    """
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    
    # Remove duplicates and sort
    sorted_indices = np.argsort(fprs)
    fprs_sorted = fprs[sorted_indices]
    tprs_sorted = tprs[sorted_indices]
    
    # Remove duplicate FPR values (keep first occurrence)
    unique_mask = np.concatenate([[True], fprs_sorted[1:] != fprs_sorted[:-1]])
    fprs_unique = fprs_sorted[unique_mask]
    tprs_unique = tprs_sorted[unique_mask]
    
    # Ensure we have enough points for interpolation
    if len(fprs_unique) < 2:
        return 0.0
    
    # Clip to valid range
    if target_fpr < fprs_unique[0]:
        return tprs_unique[0]
    if target_fpr > fprs_unique[-1]:
        return tprs_unique[-1]
    
    # Linear interpolation
    tpr_interp = np.interp(target_fpr, fprs_unique, tprs_unique)
    
    return float(tpr_interp)


def create_tpr_bar_chart(results, target_fprs=[0.01, 0.03, 0.1], 
                         output_file="evaluation/figures/tpr_bars.png"):
    """
    Create bar chart showing TPR at specific FPR values for all configurations.
    
    Args:
        results: List of attack result dictionaries
        target_fprs: List of FPR values to evaluate at
        output_file: Path to save the plot
    """
    if not results:
        print("No results to plot!")
        return
    
    # Organize results by algorithm and attack type
    organized = {}
    
    for result in results:
        if "roc_curve" not in result:
            continue
        
        algo = result.get("algorithm", "unknown")
        attack_type = result.get("attack_type", "unknown")
        hyperparams = result.get("hyperparameters", {})
        
        # Create label
        if algo == "baseline":
            label = f"Baseline"
            sort_key = (0, label)
        elif algo == "dpsgd":
            noise = hyperparams.get("noise_multiplier", "?")
            label = f"DP-SGD σ={noise}"
            sort_key = (1, noise)
        elif algo == "genericbbl":
            epsilon = hyperparams.get("epsilon", "?")
            label = f"GenericBBL ε={epsilon}"
            sort_key = (2, epsilon)
        else:
            label = algo
            sort_key = (3, label)
        
        key = (algo, attack_type, label, sort_key)
        
        # Get TPR at each target FPR
        fprs = result["roc_curve"]["fprs"]
        tprs = result["roc_curve"]["tprs"]
        
        tpr_values = []
        for target_fpr in target_fprs:
            tpr = get_tpr_at_fpr(fprs, tprs, target_fpr)
            tpr_values.append(tpr)
        
        organized[key] = tpr_values
    
    # Sort by algorithm and configuration
    sorted_keys = sorted(organized.keys(), key=lambda x: (x[3], x[1]))
    
    # Prepare data for plotting
    labels = []
    attack_types = []
    data_by_fpr = {fpr: [] for fpr in target_fprs}
    
    for key in sorted_keys:
        algo, attack_type, label, sort_key = key
        labels.append(f"{label}\n{attack_type}")
        attack_types.append(attack_type)
        
        tpr_values = organized[key]
        for i, fpr in enumerate(target_fprs):
            data_by_fpr[fpr].append(tpr_values[i])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(labels))
    width = 0.25  # Width of bars
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Plot bars for each FPR
    for i, (fpr, color) in enumerate(zip(target_fprs, colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data_by_fpr[fpr], width, 
                     label=f'FPR = {fpr}', color=color, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only show label if bar is visible
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=7, rotation=0)
    
    # Formatting
    ax.set_xlabel('Algorithm Configuration & Attack Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    ax.set_title('Membership Inference Attack Performance: TPR at Different FPR Thresholds', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, min(1.0, max([max(v) for v in data_by_fpr.values()]) * 1.15))
    
    # Add vertical lines to separate algorithm groups
    prev_algo = None
    for i, key in enumerate(sorted_keys):
        algo = key[0]
        if prev_algo is not None and algo != prev_algo:
            ax.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        prev_algo = algo
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Bar chart saved to: {output_file}")
    plt.close()


def create_grouped_bar_charts(results, target_fprs=[0.01, 0.03, 0.1],
                              output_dir="evaluation/figures"):
    """
    Create separate bar charts for each FPR threshold.
    
    Args:
        results: List of attack result dictionaries
        target_fprs: List of FPR values to evaluate at
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for target_fpr in target_fprs:
        # Organize results
        organized = {}
        
        for result in results:
            if "roc_curve" not in result:
                continue
            
            algo = result.get("algorithm", "unknown")
            attack_type = result.get("attack_type", "unknown")
            hyperparams = result.get("hyperparameters", {})
            
            # Create label
            if algo == "baseline":
                label = f"Baseline"
                sort_key = (0, 0)
            elif algo == "dpsgd":
                noise = hyperparams.get("noise_multiplier", 0)
                label = f"DP-SGD\nσ={noise}"
                sort_key = (1, noise)
            elif algo == "genericbbl":
                epsilon = hyperparams.get("epsilon", 0)
                label = f"GenericBBL\nε={epsilon}"
                sort_key = (2, epsilon)
            else:
                label = algo
                sort_key = (3, 0)
            
            key = (algo, attack_type, label, sort_key)
            
            # Get TPR at target FPR
            fprs = result["roc_curve"]["fprs"]
            tprs = result["roc_curve"]["tprs"]
            tpr = get_tpr_at_fpr(fprs, tprs, target_fpr)
            
            organized[key] = tpr
        
        # Sort and prepare data
        sorted_keys = sorted(organized.keys(), key=lambda x: (x[3], x[1]))
        
        labels = []
        tpr_values = []
        colors_list = []
        
        for key in sorted_keys:
            algo, attack_type, label, sort_key = key
            labels.append(f"{label}\n({attack_type})")
            tpr_values.append(organized[key])
            
            # Color by algorithm
            if algo == "baseline":
                colors_list.append('#1f77b4')  # Blue
            elif algo == "dpsgd":
                colors_list.append('#ff7f0e')  # Orange
            elif algo == "genericbbl":
                colors_list.append('#2ca02c')  # Green
            else:
                colors_list.append('gray')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(labels))
        bars = ax.bar(x, tpr_values, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Algorithm Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
        ax.set_title(f'Membership Inference Attack Performance at FPR = {target_fpr}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, min(1.0, max(tpr_values) * 1.15) if tpr_values else 1.0)
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.8, edgecolor='black', label='Baseline'),
            Patch(facecolor='#ff7f0e', alpha=0.8, edgecolor='black', label='DP-SGD'),
            Patch(facecolor='#2ca02c', alpha=0.8, edgecolor='black', label='GenericBBL')
        ]
        ax.legend(handles=legend_elements, fontsize=11, loc='upper left')
        
        plt.tight_layout()
        
        output_file = output_path / f"tpr_bars_fpr{target_fpr:.2f}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Bar chart saved to: {output_file}")
        plt.close()


def print_tpr_table(results, target_fprs=[0.01, 0.03, 0.1]):
    """
    Print a table of TPR values at specific FPR thresholds.
    
    Args:
        results: List of attack result dictionaries
        target_fprs: List of FPR values to evaluate at
    """
    print("\n" + "=" * 100)
    print("TPR Values at Specific FPR Thresholds")
    print("=" * 100)
    
    # Organize results
    data = []
    
    for result in results:
        if "roc_curve" not in result:
            continue
        
        algo = result.get("algorithm", "unknown")
        attack_type = result.get("attack_type", "unknown")
        hyperparams = result.get("hyperparameters", {})
        
        # Create label
        if algo == "baseline":
            config = "Baseline"
            sort_key = (0, 0)
        elif algo == "dpsgd":
            noise = hyperparams.get("noise_multiplier", 0)
            config = f"σ={noise}"
            sort_key = (1, noise)
        elif algo == "genericbbl":
            epsilon = hyperparams.get("epsilon", 0)
            config = f"ε={epsilon}"
            sort_key = (2, epsilon)
        else:
            config = "unknown"
            sort_key = (3, 0)
        
        # Get TPR at each target FPR
        fprs = result["roc_curve"]["fprs"]
        tprs = result["roc_curve"]["tprs"]
        
        tpr_values = {}
        for target_fpr in target_fprs:
            tpr = get_tpr_at_fpr(fprs, tprs, target_fpr)
            tpr_values[target_fpr] = tpr
        
        data.append((sort_key, algo, config, attack_type, tpr_values))
    
    # Sort and print
    data.sort(key=lambda x: (x[0], x[3]))
    
    # Print header
    header = f"{'Algorithm':<15} {'Config':<15} {'Attack Type':<15}"
    for fpr in target_fprs:
        header += f" {'TPR@' + f'{fpr}':<12}"
    print(header)
    print("-" * 100)
    
    # Print data
    for _, algo, config, attack_type, tpr_values in data:
        row = f"{algo:<15} {config:<15} {attack_type:<15}"
        for fpr in target_fprs:
            row += f" {tpr_values[fpr]:<12.4f}"
        print(row)
    
    print("=" * 100)


def main():
    """Main function to generate TPR bar charts."""
    print("=" * 80)
    print("Generating TPR Bar Charts at Specific FPR Thresholds")
    print("=" * 80)
    
    # Load results
    print("\nLoading results...")
    results = load_all_results("evaluation/data")
    
    if not results:
        print("No results found! Run attacks first.")
        return
    
    print(f"Loaded {len(results)} attack results")
    
    # Define FPR thresholds
    target_fprs = [0.01, 0.03, 0.1]
    
    # Print table
    print_tpr_table(results, target_fprs)
    
    # Create plots
    print("\nGenerating bar charts...")
    print("-" * 80)
    
    # Combined bar chart
    create_tpr_bar_chart(results, target_fprs)
    
    # Individual bar charts per FPR
    create_grouped_bar_charts(results, target_fprs)
    
    print("\n" + "=" * 80)
    print("All bar charts generated successfully!")
    print("=" * 80)
    print("\nPlots saved in: evaluation/figures/")


if __name__ == "__main__":
    main()

