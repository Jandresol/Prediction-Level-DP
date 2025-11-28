"""
Extract Top Vulnerable Samples and Find Intersection Across Experiments

This script analyzes membership inference attack results, extracts the top N most
vulnerable samples from each experiment, and finds the intersection of vulnerable
samples across all experiments.
"""

import json
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def extract_vulnerable_samples_intersection(
    results_dir="results",
    num_models_filter=1000,
    top_n=500,
    output_dir="results/vulnerable_samples"
):
    """
    Extract top N vulnerable samples from each experiment and find intersection.
    
    Args:
        results_dir: Directory containing experiment results
        num_models_filter: Only process experiments with this many models
        top_n: Number of top vulnerable samples to extract
        output_dir: Directory to save results
    
    Returns:
        Dictionary with vulnerable sample sets and intersection
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TPR JSON files with the specified number of models
    pattern = f"*_tprs_{num_models_filter}models_*.json"
    tpr_files = list(results_dir.glob(pattern))
    
    if not tpr_files:
        print(f"No experiment files found matching pattern: {pattern}")
        return None
    
    print(f"Found {len(tpr_files)} experiments with {num_models_filter} models:")
    for f in tpr_files:
        print(f"  - {f.name}")
    
    print(f"\n{'=' * 80}")
    print(f"Extracting top {top_n} vulnerable sample IDs from each experiment...")
    print('=' * 80)
    
    vulnerable_sets = {}
    experiment_data = {}
    
    for tpr_file in tpr_files:
        experiment_name = tpr_file.stem  # Filename without extension
        print(f"\n{experiment_name}:")
        
        # Load TPR data
        with open(tpr_file, 'r') as f:
            tprs = json.load(f)
        
        num_samples = len(tprs)
        print(f"  Total samples: {num_samples}")
        
        # Convert to numpy array
        tprs_array = np.array(tprs)
        
        # Get indices sorted by TPR (descending)
        sorted_indices = np.argsort(tprs_array)[::-1]
        
        # Extract top N vulnerable sample IDs
        top_indices = sorted_indices[:top_n]
        top_tprs = tprs_array[top_indices]
        
        # Store as set for intersection computation
        vulnerable_sets[experiment_name] = set(top_indices.tolist())
        
        # Store detailed data
        experiment_data[experiment_name] = {
            "top_indices": top_indices.tolist(),
            "top_tprs": top_tprs.tolist(),
            "stats": {
                "mean": float(np.mean(top_tprs)),
                "median": float(np.median(top_tprs)),
                "min": float(np.min(top_tprs)),
                "max": float(np.max(top_tprs)),
                "std": float(np.std(top_tprs))
            }
        }
        
        print(f"  Top {top_n} TPR stats: mean={np.mean(top_tprs):.4f}, "
              f"median={np.median(top_tprs):.4f}, max={np.max(top_tprs):.4f}")
    
    # Compute intersection
    print(f"\n{'=' * 80}")
    print("Computing intersection of vulnerable samples...")
    print('=' * 80)
    
    if len(vulnerable_sets) == 0:
        print("No experiments found!")
        return None
    
    # Get intersection across all experiments
    experiment_names = list(vulnerable_sets.keys())
    intersection = vulnerable_sets[experiment_names[0]].copy()
    
    for exp_name in experiment_names[1:]:
        intersection = intersection.intersection(vulnerable_sets[exp_name])
    
    intersection_list = sorted(list(intersection))
    
    print(f"\nIntersection results:")
    print(f"  Number of experiments: {len(vulnerable_sets)}")
    print(f"  Top N per experiment: {top_n}")
    print(f"  Intersection size: {len(intersection_list)}")
    print(f"  Percentage: {100 * len(intersection_list) / top_n:.2f}%")
    
    # Compute pairwise intersections
    pairwise_intersections = {}
    print(f"\nPairwise intersection sizes:")
    for i, exp1 in enumerate(experiment_names):
        for exp2 in experiment_names[i+1:]:
            inter = vulnerable_sets[exp1].intersection(vulnerable_sets[exp2])
            key = f"{exp1} ∩ {exp2}"
            pairwise_intersections[key] = len(inter)
            # Shortened names for display
            exp1_short = exp1.replace(f"_tprs_{num_models_filter}models", "")
            exp2_short = exp2.replace(f"_tprs_{num_models_filter}models", "")
            print(f"  {exp1_short} ∩ {exp2_short}: {len(inter)} samples ({100*len(inter)/top_n:.1f}%)")
    
    # Prepare results
    results = {
        "num_models": num_models_filter,
        "top_n": top_n,
        "num_experiments": len(vulnerable_sets),
        "experiments": list(experiment_names),
        "intersection": {
            "sample_ids": intersection_list,
            "size": len(intersection_list),
            "percentage": 100 * len(intersection_list) / top_n
        },
        "pairwise_intersections": pairwise_intersections,
        "per_experiment": experiment_data
    }
    
    # Save results
    output_file = output_dir / f"vulnerable_intersection_top{top_n}_{num_models_filter}models.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Save just the intersection IDs for easy use
    intersection_file = output_dir / f"intersection_ids_top{top_n}_{num_models_filter}models.txt"
    with open(intersection_file, 'w') as f:
        for idx in intersection_list:
            f.write(f"{idx}\n")
    print(f"Intersection IDs saved to: {intersection_file}")
    
    # Create visualizations
    create_visualizations(results, output_dir, top_n, num_models_filter)
    
    return results


def create_visualizations(results, output_dir, top_n, num_models):
    """
    Create visualizations for vulnerable sample analysis.
    
    Args:
        results: Dictionary with analysis results
        output_dir: Directory to save plots
        top_n: Number of top samples
        num_models: Number of models in experiments
    """
    experiment_names = results["experiments"]
    per_experiment = results["per_experiment"]
    intersection_ids = set(results["intersection"]["sample_ids"])
    
    # Plot 1: Venn-style diagram showing set sizes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart of set sizes
    ax = axes[0]
    set_sizes = []
    labels = []
    
    for exp_name in experiment_names:
        set_sizes.append(len(per_experiment[exp_name]["top_indices"]))
        label = exp_name.replace(f"_tprs_{num_models}models", "").replace("_", "\n")
        labels.append(label)
    
    set_sizes.append(results["intersection"]["size"])
    labels.append("Intersection")
    
    colors = ['steelblue'] * len(experiment_names) + ['orange']
    bars = ax.bar(range(len(set_sizes)), set_sizes, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title(f"Vulnerable Sample Set Sizes\n(Top {top_n} from each experiment)", fontsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=top_n, color='red', linestyle='--', alpha=0.5, label=f'Top {top_n}')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: TPR distribution for intersection vs non-intersection
    ax = axes[1]
    
    for exp_name in experiment_names:
        indices = per_experiment[exp_name]["top_indices"]
        tprs = per_experiment[exp_name]["top_tprs"]
        
        # Separate into intersection and non-intersection
        in_intersection = [tpr for idx, tpr in zip(indices, tprs) if idx in intersection_ids]
        not_in_intersection = [tpr for idx, tpr in zip(indices, tprs) if idx not in intersection_ids]
        
        label = exp_name.replace(f"_tprs_{num_models}models", "").replace("_", " ")
        
        if in_intersection:
            ax.scatter([label] * len(in_intersection), in_intersection, 
                      alpha=0.5, s=20, color='red', label='In intersection' if exp_name == experiment_names[0] else '')
        if not_in_intersection:
            ax.scatter([label] * len(not_in_intersection), not_in_intersection, 
                      alpha=0.3, s=10, color='blue', label='Not in intersection' if exp_name == experiment_names[0] else '')
    
    ax.set_ylabel("TPR", fontsize=11)
    ax.set_title(f"TPR Distribution: Intersection vs Non-Intersection\n(Top {top_n} samples)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / f"vulnerable_analysis_top{top_n}_{num_models}models.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_file}")
    
    # Plot 3: Pairwise intersection heatmap (if multiple experiments)
    if len(experiment_names) > 1:
        create_intersection_heatmap(results, output_dir, top_n, num_models)


def create_intersection_heatmap(results, output_dir, top_n, num_models):
    """
    Create a heatmap showing pairwise intersections between experiments.
    
    Args:
        results: Dictionary with analysis results
        output_dir: Directory to save plot
        top_n: Number of top samples
        num_models: Number of models
    """
    experiment_names = results["experiments"]
    per_experiment = results["per_experiment"]
    n = len(experiment_names)
    
    # Create intersection matrix
    intersection_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                intersection_matrix[i, j] = top_n
            else:
                set_i = set(per_experiment[experiment_names[i]]["top_indices"])
                set_j = set(per_experiment[experiment_names[j]]["top_indices"])
                intersection_matrix[i, j] = len(set_i.intersection(set_j))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(intersection_matrix, cmap='YlOrRd', aspect='auto')
    
    # Labels
    short_names = [name.replace(f"_tprs_{num_models}models", "").replace("_", "\n") 
                   for name in experiment_names]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values to cells
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{int(intersection_matrix[i, j])}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title(f"Pairwise Intersection of Top {top_n} Vulnerable Samples", fontsize=12, pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Shared Samples', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    output_file = output_dir / f"intersection_heatmap_top{top_n}_{num_models}models.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {output_file}")


def main():
    """Main function to extract and analyze vulnerable samples."""
    # Configuration
    results_dir = "results"
    num_models_filter = 1000  # Only process experiments with 1000 models
    top_n = 500  # Extract top 500 vulnerable samples
    output_dir = "results/vulnerable_samples"
    
    print("=" * 80)
    print("Extracting Vulnerable Samples and Computing Intersection")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Filtering experiments with: {num_models_filter} models")
    print(f"Extracting top: {top_n} vulnerable samples per experiment")
    print(f"Output directory: {output_dir}")
    print()
    
    results = extract_vulnerable_samples_intersection(
        results_dir=results_dir,
        num_models_filter=num_models_filter,
        top_n=top_n,
        output_dir=output_dir
    )
    
    if results:
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)
        print(f"\nProcessed {results['num_experiments']} experiments")
        print(f"Intersection contains {results['intersection']['size']} samples")
        print(f"Output saved in: {output_dir}/")


if __name__ == "__main__":
    main()

