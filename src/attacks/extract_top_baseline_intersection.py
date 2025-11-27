"""
Extract Top 100 Most Vulnerable Samples from Baseline Experiments Intersection

This script:
1. Finds the intersection of top 500 from both baseline experiments
2. Ranks each sample by its worst (highest) ranking across both experiments
   - This ensures we select samples that are consistently vulnerable in BOTH experiments
   - e.g., (1, 100) is placed after (2, 2) because max(1,100)=100 > max(2,2)=2
3. Selects the top 100 most vulnerable samples
4. Saves the result to a text file
"""

import json
import numpy as np
from pathlib import Path


def extract_top_baseline_intersection(
    intersection_file="results/vulnerable_samples/vulnerable_intersection_top500_1000models.json",
    output_file="results/vulnerable_samples/top100_baseline_intersection.txt",
    top_n=100
):
    """
    Extract top N most vulnerable samples from baseline experiments intersection.
    Samples are ranked by their worst (highest) ranking across both experiments,
    ensuring consistent vulnerability in both.
    
    Args:
        intersection_file: Path to the intersection JSON file
        output_file: Path to save the top sample IDs
        top_n: Number of top samples to extract
    """
    print("=" * 80)
    print("Extracting Top Vulnerable Samples from Baseline Experiments Intersection")
    print("=" * 80)
    
    # Load intersection data
    print(f"\nLoading data from: {intersection_file}")
    with open(intersection_file, 'r') as f:
        data = json.load(f)
    
    # Find the two baseline experiments
    baseline_experiments = [exp for exp in data['experiments'] 
                           if 'baseline' in exp.lower() and 'dp' not in exp.lower()]
    
    if len(baseline_experiments) != 2:
        print(f"Error: Expected 2 baseline experiments, found {len(baseline_experiments)}")
        print(f"Baseline experiments found: {baseline_experiments}")
        return
    
    exp1_name = baseline_experiments[0]
    exp2_name = baseline_experiments[1]
    
    print(f"\nBaseline experiments:")
    print(f"  1. {exp1_name}")
    print(f"  2. {exp2_name}")
    
    # Get top indices from both experiments
    exp1_indices = data['per_experiment'][exp1_name]['top_indices']
    exp2_indices = data['per_experiment'][exp2_name]['top_indices']
    
    print(f"\nExperiment 1: {len(exp1_indices)} samples")
    print(f"Experiment 2: {len(exp2_indices)} samples")
    
    # Find intersection
    exp1_set = set(exp1_indices)
    exp2_set = set(exp2_indices)
    intersection = exp1_set.intersection(exp2_set)
    
    print(f"\nIntersection: {len(intersection)} samples")
    
    if len(intersection) == 0:
        print("No samples in intersection!")
        return
    
    # For each sample in intersection, get its ranking in both experiments
    sample_rankings = []
    
    for sample_id in intersection:
        # Ranking is the position in the list (0 = most vulnerable)
        rank1 = exp1_indices.index(sample_id)
        rank2 = exp2_indices.index(sample_id)
        
        # Take the worse (higher) ranking - we want samples consistently vulnerable in BOTH
        worst_rank = max(rank1, rank2)
        
        sample_rankings.append({
            'sample_id': sample_id,
            'rank1': rank1,
            'rank2': rank2,
            'worst_rank': worst_rank
        })
    
    # Sort by worst ranking (lower worst rank means more consistent vulnerability)
    sample_rankings.sort(key=lambda x: x['worst_rank'])
    
    # Take top N
    top_samples = sample_rankings[:top_n]
    
    print(f"\nTop {top_n} most vulnerable samples (by worst ranking - consistent in both):")
    print(f"{'Rank':<6} {'Sample ID':<12} {'Rank in Exp1':<15} {'Rank in Exp2':<15} {'Worst Rank':<12}")
    print("-" * 70)
    
    for i, sample in enumerate(top_samples[:20], 1):  # Show first 20
        print(f"{i:<6} {sample['sample_id']:<12} {sample['rank1']:<15} "
              f"{sample['rank2']:<15} {sample['worst_rank']:<12}")
    
    if len(top_samples) > 20:
        print(f"... (showing first 20 of {len(top_samples)})")
    
    # Get statistics
    worst_ranks = [s['worst_rank'] for s in top_samples]
    rank1_vals = [s['rank1'] for s in top_samples]
    rank2_vals = [s['rank2'] for s in top_samples]
    
    print(f"\nStatistics for top {top_n} samples:")
    print(f"  Worst rank range: [{min(worst_ranks)}, {max(worst_ranks)}]")
    print(f"  Mean worst rank: {np.mean(worst_ranks):.2f}")
    print(f"  Median worst rank: {np.median(worst_ranks):.2f}")
    print(f"  Exp1 rank range: [{min(rank1_vals)}, {max(rank1_vals)}]")
    print(f"  Exp2 rank range: [{min(rank2_vals)}, {max(rank2_vals)}]")
    
    # Count how many came primarily from each experiment
    worse_in_exp1 = sum(1 for s in top_samples if s['rank1'] > s['rank2'])
    worse_in_exp2 = sum(1 for s in top_samples if s['rank2'] > s['rank1'])
    tied = sum(1 for s in top_samples if s['rank1'] == s['rank2'])
    
    print(f"\nWorst ranking source:")
    print(f"  Worse in Exp1: {worse_in_exp1} samples")
    print(f"  Worse in Exp2: {worse_in_exp2} samples")
    print(f"  Tied: {tied} samples")
    
    # Save to text file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in top_samples:
            f.write(f"{sample['sample_id']}\n")
    
    print(f"\n✓ Saved top {len(top_samples)} sample IDs to: {output_file}")
    
    # Also save detailed version with rankings
    detailed_file = output_path.parent / output_path.name.replace('.txt', '_detailed.json')
    detailed_data = {
        'description': f'Top {top_n} most vulnerable samples from baseline experiments intersection (sorted by worst rank - consistent in both)',
        'baseline_experiments': baseline_experiments,
        'intersection_size': len(intersection),
        'top_n': top_n,
        'samples': top_samples,
        'statistics': {
            'worst_rank_range': [int(min(worst_ranks)), int(max(worst_ranks))],
            'mean_worst_rank': float(np.mean(worst_ranks)),
            'median_worst_rank': float(np.median(worst_ranks)),
            'worse_in_exp1': worse_in_exp1,
            'worse_in_exp2': worse_in_exp2,
            'tied': tied
        }
    }
    
    with open(detailed_file, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    
    print(f"✓ Saved detailed data to: {detailed_file}")
    
    print("\n" + "=" * 80)
    print("Extraction Complete!")
    print("=" * 80)
    
    return top_samples


def main():
    """Main function."""
    extract_top_baseline_intersection(
        intersection_file="results/vulnerable_samples/vulnerable_intersection_top500_1000models.json",
        output_file="results/vulnerable_samples/top100_baseline_intersection.txt",
        top_n=100
    )


if __name__ == "__main__":
    main()

