import json
import os
import matplotlib.pyplot as plt
import sys

from matplotlib.ticker import PercentFormatter
def plot_privacy_utility_tradeoff():
    """
    Reads the JSON data from the accuracy evaluation summary and plots
    the privacy-utility trade-off curve for both GenericBBL and DPSGD.
    """
    # Define paths relative to the project root
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(ROOT, "results", "metrics", "accuracy_evaluation_summary.json")
    # Save plot in the evaluation/figures directory
    plot_dir = os.path.join(ROOT, "evaluation", "figures")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "privacy_utility_tradeoff.png")

    # Ensure the script can be run from any directory
    sys.path.insert(0, ROOT)

    if not os.path.exists(json_path):
        print(f"Error: Data file not found at {json_path}")
        print("Please run `experiments/evaluate_accuracy.py` first to generate the data.")
        return

    # Read the data from the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data:
        print("Error: The data file is empty. No data to plot.")
        return

    # Separate data by algorithm
    dpsgd_data = [item for item in data if item['algorithm'] == 'dpsgd']
    genericbbl_data = [item for item in data if item['algorithm'] == 'genericbbl']

    # --- Process and Sort DPSGD data ---
    if dpsgd_data:
        epsilons_dpsgd = [item['final_epsilon'] for item in dpsgd_data]
        accuracies_dpsgd = [item['accuracy'] for item in dpsgd_data]
        sorted_pairs_dpsgd = sorted(zip(epsilons_dpsgd, accuracies_dpsgd))
        epsilons_dpsgd_sorted, accuracies_dpsgd_sorted = zip(*sorted_pairs_dpsgd)

    # --- Process and Sort GenericBBL data ---
    if genericbbl_data:
        epsilons_bbl = [item['final_epsilon'] for item in genericbbl_data]
        accuracies_bbl = [item['accuracy'] for item in genericbbl_data]
        sorted_pairs_bbl = sorted(zip(epsilons_bbl, accuracies_bbl))
        epsilons_bbl_sorted, accuracies_bbl_sorted = zip(*sorted_pairs_bbl)


    # --- Create the Plot ---
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 6))
    
    # Plot DPSGD data
    if dpsgd_data:
        plt.plot(epsilons_dpsgd_sorted, accuracies_dpsgd_sorted, marker='o', linestyle='-', color='blue', label='DPSGD')

    # Plot GenericBBL data
    if genericbbl_data:
        plt.plot(epsilons_bbl_sorted, accuracies_bbl_sorted, marker='s', linestyle='--', color='green', label='GenericBBL')

    # Add titles and labels for clarity
    plt.title('Privacy-Utility Trade-off on CIFAR-10', fontsize=16)
    plt.xlabel('Epsilon (Privacy Budget)', fontsize=12)
    plt.ylabel('Test Accuracy (Utility)', fontsize=12)
    
    # Set axis limits and add a legend
    plt.ylim(0.5, 1.0)
    plt.xlim(0, 180)
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
    plot_privacy_utility_tradeoff()
