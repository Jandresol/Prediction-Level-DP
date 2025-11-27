"""
Visualize Top Vulnerable Samples from CIFAR-10

This script loads the top vulnerable sample IDs and displays their images.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, '.')

from src.datasets.load_cifar10 import load_torch_dataset


def visualize_vulnerable_samples(
    sample_ids_file="results/vulnerable_samples/top100_baseline_intersection.txt",
    output_file="results/vulnerable_samples/top10_vulnerable_images.png",
    num_samples=10
):
    """
    Visualize the top N vulnerable sample images.
    
    Args:
        sample_ids_file: Path to text file with sample IDs
        output_file: Path to save the visualization
        num_samples: Number of samples to visualize
    """
    print("=" * 80)
    print("Visualizing Top Vulnerable Samples")
    print("=" * 80)
    
    # Load sample IDs
    print(f"\nLoading sample IDs from: {sample_ids_file}")
    with open(sample_ids_file, 'r') as f:
        sample_ids = [int(line.strip()) for line in f if line.strip()]
    
    print(f"Total sample IDs: {len(sample_ids)}")
    print(f"Top {num_samples} sample IDs: {sample_ids[:num_samples]}")
    
    # Load CIFAR-10 binary dataset
    print(f"\nLoading CIFAR-10 binary dataset...")
    train_data, test_data = load_torch_dataset("cifar10_binary")
    
    images = train_data["images"]
    labels = train_data["labels"]
    label_names = train_data["label_names"]
    
    print(f"Dataset loaded: {len(images)} images")
    print(f"Label names: {label_names}")
    
    # Select top N samples
    top_sample_ids = sample_ids[:num_samples]
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Top {num_samples} Most Vulnerable Samples\n(Consistently vulnerable in both baseline experiments)', 
                 fontsize=14, fontweight='bold')
    
    for idx, sample_id in enumerate(top_sample_ids):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        # Get image and label
        # Images are in CHW format (uint8), convert to HWC for display
        image = images[sample_id].permute(1, 2, 0).numpy()
        label_idx = labels[sample_id].item()
        label_name = label_names[label_idx]
        
        # Display image
        ax.imshow(image)
        ax.axis('off')
        
        # Set title with sample ID and label
        ax.set_title(f'ID: {sample_id}\n{label_name} (Label {label_idx})', 
                    fontsize=10, fontweight='bold')
        
        # Add border to make it stand out
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    
    # Also display
    plt.show()
    
    # Print detailed info
    print("\nDetailed information:")
    print(f"{'Rank':<6} {'Sample ID':<12} {'Label':<15} {'Label Name':<15}")
    print("-" * 50)
    for rank, sample_id in enumerate(top_sample_ids, 1):
        label_idx = labels[sample_id].item()
        label_name = label_names[label_idx]
        print(f"{rank:<6} {sample_id:<12} {label_idx:<15} {label_name:<15}")
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)


def main():
    """Main function."""
    visualize_vulnerable_samples(
        sample_ids_file="results/vulnerable_samples/top100_baseline_intersection.txt",
        output_file="results/vulnerable_samples/top10_vulnerable_images.png",
        num_samples=10
    )


if __name__ == "__main__":
    main()

