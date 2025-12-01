import json
import os
import matplotlib.pyplot as plt
import sys

from matplotlib.ticker import PercentFormatter
def plot_epsilon_vs_accuracy():
    """
    Reads the JSON data from the epsilon-accuracy experiment and plots
    the privacy-utility trade-off curve for GenericBBL.
    """
    # Define paths relative to the project root
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(ROOT, "results", "metrics", "genericbbl_epsilon_vs_accuracy.json")
    plot_path = os.path.join(ROOT, "results", "genericbbl_epsilon_vs_accuracy.png")

    # Ensure the script can be run from any directory
    sys.path.insert(0, ROOT)

    if not os.path.exists(json_path):
        print(f"Error: Data file not found at {json_path}")
        print("Please run `experiments/run_genericbbl_experiment.py` first to generate the data.")
        return

    # Read the data from the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data:
        print("Error: The data file is empty. No data to plot.")
        return

    # Extract epsilon and accuracy values for plotting
    epsilons = [item['epsilon'] for item in data]
    accuracies = [item['accuracy'] for item in data]

    # Sort the data by epsilon for a clean plot
    sorted_pairs = sorted(zip(epsilons, accuracies))
    epsilons_sorted, accuracies_sorted = zip(*sorted_pairs)

    # --- Create the Plot ---
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 6))
    
    # Plot data points and the line connecting them
    plt.plot(epsilons_sorted, accuracies_sorted, marker='o', linestyle='-', color='b', label='GenericBBL Accuracy')

    # Use a logarithmic scale for the x-axis due to the wide range of epsilon values
    plt.xscale('log')
    
    # Add titles and labels for clarity
    plt.title('GenericBBL: Privacy-Utility Trade-off on CIFAR-10', fontsize=16)
    plt.xlabel('Epsilon (Privacy Budget)', fontsize=12)
    plt.ylabel('Test Accuracy (Utility)', fontsize=12)
    
    # Set axis limits and add a legend
    plt.ylim(0.0, 1.0)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.legend(fontsize=12)
    
    # Improve tick readability
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Save the figure to the results directory
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    plot_epsilon_vs_accuracy()
