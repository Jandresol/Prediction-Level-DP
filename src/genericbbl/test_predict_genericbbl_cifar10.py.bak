import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
import sys

# Add project root to path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.datasets.load_cifar10 import load_torch_dataset
from src.genericbbl.predict_genericbbl import PrivateEverlastingPredictor

def test_genericbbl_on_cifar10():
    """
    Test the PrivateEverlastingPredictor on the binary CIFAR-10 dataset.
    """
    print("--- Loading CIFAR-10 Binary Dataset ---")
    
    # 1. Load the pre-processed binary CIFAR-10 dataset
    # This dataset is located in 'cifar10_binary/' and contains airplanes vs. automobiles
    try:
        train_data, test_data = load_torch_dataset("cifar10_binary")
    except FileNotFoundError:
        print("\nError: 'cifar10_binary' dataset not found.")
        print("Please run 'python src/datasets/load_cifar10.py' to generate it first.")
        return

    # Combine train and test to form a larger pool of data
    X_all = torch.cat([train_data['images'], test_data['images']], dim=0)
    y_all = torch.cat([train_data['labels'], test_data['labels']], dim=0)

    # Convert to numpy and flatten images for scikit-learn
    # Images are [N, 3, 32, 32], flatten to [N, 3072]
    X_all_np = X_all.numpy().reshape(X_all.shape[0], -1)
    y_all_np = y_all.numpy()

    # Normalize pixel values to [0, 1]
    X_all_np = X_all_np.astype(np.float32) / 255.0

    print(f"Total dataset size: {len(X_all_np)} samples")

    # 2. Split data into an initial labeled set (S) and a query stream
    initial_size = 1000
    X_initial, X_stream = X_all_np[:initial_size], X_all_np[initial_size:]
    y_initial, y_stream = y_all_np[:initial_size], y_all_np[initial_size:]
    print(f"Initial training set size: {len(X_initial)}")
    print(f"Query stream size: {len(X_stream)}")

    # 3. Initialize the PrivateEverlastingPredictor
    # The VC dimension of a decision tree is complex. We use an estimate.
    # Using practical_mode=True is crucial for running this on standard hardware.
    print("\n--- Initializing PrivateEverlastingPredictor ---")
    predictor = PrivateEverlastingPredictor(
        base_learner=DecisionTreeClassifier(max_depth=5),
        vc_dim=20,  # Estimated VC-dim for a decision tree of this complexity
        alpha=0.25,
        beta=0.1,
        practical_mode=True
    )
    predictor.auto_set_epsilon(X_initial.shape[0])

    # 4. Perform initial training on the set S
    predictor.train_initial(X_initial, y_initial)

    # 5. Simulate the everlasting process by running rounds on chunks of the stream
    print("\n--- Starting Prediction Rounds ---")
    chunk_size = 500  # Number of queries to process per round
    for i in range(0, len(X_stream), chunk_size):
        batch = X_stream[i : i + chunk_size]
        
        if len(batch) == 0:
            break
        
        predictions = predictor.run_round(batch)
        
        # Evaluate utility by checking accuracy on the stream chunk
        true_labels = y_stream[i : i + len(predictions)]
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Stream Chunk {i//chunk_size + 1} Accuracy: {accuracy:.3f} on {len(predictions)} samples")

if __name__ == "__main__":
    test_genericbbl_on_cifar10()
