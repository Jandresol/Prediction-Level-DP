import torch
from sklearn.tree import DecisionTreeClassifier
from src.models.cifar_cnn import cifar_cnn
from src.genericbbl.pytorch_wrapper import PyTorchCNNWrapper
from sklearn.metrics import accuracy_score
import numpy as np
import os
import sys
import urllib.request
import tarfile
import subprocess

# Add project root to path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.datasets.load_cifar10 import load_torch_dataset
from src.genericbbl.predict_genericbbl import PrivateEverlastingPredictor

from src.datasets.load_cifar10 import load_all_cifar10_as_torch

def download_cifar10_if_needed():
    """
    Download CIFAR-10 dataset to ~/cifar-10-batches-py/ if not already present.
    """
    cifar_path = os.path.expanduser('~/cifar-10-batches-py')
    
    if os.path.exists(cifar_path):
        print(f"CIFAR-10 dataset found at {cifar_path}")
        return True
    
    print("CIFAR-10 dataset not found. Attempting to download...")
    
    try:
        # Download CIFAR-10
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        tmp_path = '/tmp/cifar-10-python.tar.gz'
        extract_path = '/tmp/'
        
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, tmp_path)
        print("Download complete.")
        
        # Extract
        print("Extracting...")
        with tarfile.open(tmp_path, 'r:gz') as tar:
            tar.extractall(extract_path)
        
        # Move to home directory
        os.system(f'mv {extract_path}/cifar-10-batches-py {cifar_path}')
        print(f"CIFAR-10 dataset installed at {cifar_path}")
        
        # Clean up
        os.remove(tmp_path)
        
        return True
        
    except Exception as e:
        print(f"Error downloading CIFAR-10: {e}")
        print("\nTo download CIFAR-10 manually, run:")
        print("wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        print("tar -xzf cifar-10-python.tar.gz -C ~/")
        print("mv ~/cifar-10-batches-py ~/cifar-10-batches-py")
        return False


def ensure_cifar10_binary_dataset():
    """
    Ensure the binary CIFAR-10 dataset exists, downloading and processing if needed.
    """
    # First, download raw CIFAR-10 if needed
    if not download_cifar10_if_needed():
        return False
    
    # Check if binary dataset already exists
    if os.path.exists('cifar10_binary'):
        print("Binary CIFAR-10 dataset found at cifar10_binary/")
        return True
    
    print("Binary CIFAR-10 dataset not found. Processing raw dataset...")
    
    try:
        # Run the dataset processing script
        result = subprocess.run([
            sys.executable, 
            'src/datasets/load_cifar10.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Binary CIFAR-10 dataset created successfully.")
            return True
        else:
            print(f"Error processing dataset: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error running dataset processing: {e}")
        return False


def test_genericbbl_on_cifar10():
    """
    Test the PrivateEverlastingPredictor on the binary CIFAR-10 dataset.
    Automatically downloads and processes CIFAR-10 if needed.
    """
    print("=== CIFAR-10 GenericBBL Test ===")
    
    # --- Setup for GPU Acceleration ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # Ensure we have the binary CIFAR-10 dataset
    if not ensure_cifar10_binary_dataset():
        print("\nFailed to prepare CIFAR-10 dataset. Exiting.")
        return
    
    print("\n--- Loading CIFAR-10 Binary Dataset ---")
    

    # Load the full CIFAR-10 dataset (10 classes) without initial filtering
    try:
        data_dir = os.path.expanduser('~/cifar-10-batches-py')
        X_train, y_train, X_test, y_test, label_names = load_all_cifar10_as_torch(data_dir, normalize=False)
        
        # Normalize to [0, 1] range and move to device
        X_train = (X_train.float() / 255.0).to(device)
        X_test = (X_test.float() / 255.0).to(device)
        
        print(f"Loaded full CIFAR-10 dataset:")
        print(f"  Training: {X_train.shape[0]} images across {len(label_names)} classes")
        print(f"  Test: {X_test.shape[0]} images across {len(label_names)} classes")
        
        # For binary classification, map labels 0->0, 1->1, others->0
        # This keeps the airplane vs automobile binary task but uses more data
        y_train_binary = torch.where(y_train < 2, y_train, torch.zeros_like(y_train)).to(device)
        y_test_binary = torch.where(y_test < 2, y_test, torch.zeros_like(y_test)).to(device)
        
        # Filter to keep only airplane (0) and automobile (1) for consistency
        train_mask = y_train < 2
        test_mask = y_test < 2
        
        X_train_filtered = X_train[train_mask]
        y_train_filtered = y_train_binary[train_mask]
        X_test_filtered = X_test[test_mask]
        y_test_filtered = y_test_binary[test_mask]
        
        print(f"Filtered to binary classification (airplane vs automobile):")
        print(f"  Training: {X_train_filtered.shape[0]} images")
        print(f"  Test: {X_test_filtered.shape[0]} images")
        
        # Combine train and test to form a larger pool of data
        X_all = torch.cat([X_train_filtered, X_test_filtered], dim=0)
        y_all = torch.cat([y_train_filtered, y_test_filtered], dim=0)
        
        # Convert to numpy and flatten images for scikit-learn
        # Images are [N, 3, 32, 32], flatten to [N, 3072]
        X_all_np = X_all.cpu().numpy().reshape(X_all.shape[0], -1)
        y_all_np = y_all.cpu().numpy()
        
    except FileNotFoundError:
        print("\nError: CIFAR-10 dataset not found.")
        print("Please ensure '~/cifar-10-batches-py/' contains the CIFAR-10 data.")
        return

    # Split data into an initial labeled set (S) and a query stream
    initial_size = 10000
    X_initial, X_stream = X_all_np[:initial_size], X_all_np[initial_size:]
    y_initial, y_stream = y_all_np[:initial_size], y_all_np[initial_size:]
    print(f"\nData split for GenericBBL:")
    print(f"  Initial training set size: {len(X_initial)}")
    print(f"  Query stream size: {len(X_stream)}")

    # Initialize the PrivateEverlastingPredictor
    # The VC dimension of a decision tree is complex. We use an estimate.
    # Using practical_mode=True is crucial for running this on standard hardware.
    print("\n--- Initializing PrivateEverlastingPredictor ---")
    predictor = PrivateEverlastingPredictor(
        base_learner=PyTorchCNNWrapper(model_class=cifar_cnn, epochs=10, batch_size=128, lr=1e-3, device=device),
        vc_dim=40,  # Reduced VC-dim for smaller datasets
        delta=1e-5,
        alpha=0.2,
        beta=0.2,
        practical_mode=True,
        device=device,
    )
    predictor.auto_set_epsilon((len(X_initial),), safety_factor=5.0, force_minimum=True)

    # Perform initial training on the set S
    predictor.train_initial(X_initial, y_initial)

    # Simulate the everlasting process by running rounds on chunks of the stream
    print("\n--- Starting Prediction Rounds ---")
    chunk_size = 2000  # Number of queries to process per round
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
