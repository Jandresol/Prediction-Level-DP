"""
CIFAR-10 Binary Classification Dataset Loader

This module loads and processes CIFAR-10 dataset for binary classification tasks.
It filters the dataset to keep only two classes (airplane and automobile) and 
saves/loads the data in PyTorch format.

Dataset Details:
----------------
- Training set: 10,000 images (5,000 airplanes + 5,000 automobiles)
- Test set: 2,000 images (1,000 airplanes + 1,000 automobiles)
- Image format: torch.Size([N, 3, 32, 32]) in CHW format
- Image dtype: uint8 (pixel values in range [0, 255])
- Labels: 0 = airplane, 1 = automobile

Saved Dataset Location:
-----------------------
The processed dataset is saved in the repository at: cifar10_binary/
- train.pt: Training data
- test.pt: Test data
- metadata.json: Dataset metadata

Usage Example:
--------------
# Load the pre-processed binary dataset
from src.datasets.load_cifar10 import load_torch_dataset

train_data, test_data = load_torch_dataset('cifar10_binary')

# Access images and labels
train_images = train_data['images']  # torch.Size([10000, 3, 32, 32]), dtype=uint8
train_labels = train_data['labels']  # torch.Size([10000]), dtype=int64
label_names = train_data['label_names']  # ['airplane', 'automobile']

# Normalize images for training (if needed)
train_images_norm = train_images.float() / 255.0  # Convert to [0, 1] range

# Create DataLoader
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(train_images_norm, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

Regenerating Dataset:
---------------------
To regenerate the binary dataset from raw CIFAR-10:
1. Ensure raw CIFAR-10 data is in ~/cifar-10-batches-py/
2. Run: python src/datasets/load_cifar10.py

This will:
- Load the full CIFAR-10 dataset
- Filter to keep only labels 0 (airplane) and 1 (automobile)
- Save the filtered data in PyTorch format to cifar10_binary/
- Display verification statistics
"""

import os
import pickle
import numpy as np
from pathlib import Path
import torch


def unpickle(file_path):
    """
    Load and unpickle a CIFAR-10 batch file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the batch data
    """
    with open(file_path, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_cifar10_batch(batch_file):
    """
    Load a single CIFAR-10 batch.
    
    Args:
        batch_file: Path to the batch file
        
    Returns:
        images: numpy array of shape (10000, 32, 32, 3)
        labels: list of 10000 labels (0-9)
    """
    batch_dict = unpickle(batch_file)
    
    # Extract data and labels (keys are byte strings)
    data = batch_dict[b'data']
    labels = batch_dict[b'labels']
    
    # Reshape data from (10000, 3072) to (10000, 3, 32, 32)
    # Then transpose to (10000, 32, 32, 3) for standard image format
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return images, labels


def load_cifar10_meta(meta_file):
    """
    Load CIFAR-10 metadata containing label names.
    
    Args:
        meta_file: Path to batches.meta file
        
    Returns:
        List of label names
    """
    meta_dict = unpickle(meta_file)
    label_names = [name.decode('utf-8') for name in meta_dict[b'label_names']]
    return label_names


def load_all_cifar10(data_dir):
    """
    Load all CIFAR-10 training and test data.
    
    Args:
        data_dir: Path to cifar-10-batches-py directory
        
    Returns:
        train_images: numpy array of shape (50000, 32, 32, 3)
        train_labels: numpy array of shape (50000,)
        test_images: numpy array of shape (10000, 32, 32, 3)
        test_labels: numpy array of shape (10000,)
        label_names: list of 10 class names
    """
    data_dir = Path(data_dir)
    
    # Load training batches
    train_images_list = []
    train_labels_list = []
    
    for i in range(1, 6):
        batch_file = data_dir / f'data_batch_{i}'
        images, labels = load_cifar10_batch(batch_file)
        train_images_list.append(images)
        train_labels_list.extend(labels)
    
    train_images = np.vstack(train_images_list)
    train_labels = np.array(train_labels_list)
    
    # Load test batch
    test_batch_file = data_dir / 'test_batch'
    test_images, test_labels = load_cifar10_batch(test_batch_file)
    test_labels = np.array(test_labels)
    
    # Load metadata
    meta_file = data_dir / 'batches.meta'
    label_names = load_cifar10_meta(meta_file)
    
    return train_images, train_labels, test_images, test_labels, label_names


def filter_binary_classes(images, labels, keep_labels=[0, 1]):
    """
    Filter dataset to keep only specified labels.
    
    Args:
        images: numpy array of images
        labels: numpy array of labels
        keep_labels: list of labels to keep
        
    Returns:
        filtered_images: numpy array of filtered images
        filtered_labels: numpy array of filtered labels (remapped to 0, 1, ...)
    """
    # Create mask for samples with labels in keep_labels
    mask = np.isin(labels, keep_labels)
    
    # Filter images and labels
    filtered_images = images[mask]
    filtered_labels = labels[mask]
    
    # Remap labels to 0, 1, 2, ... (in order of keep_labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(keep_labels)}
    remapped_labels = np.array([label_mapping[label] for label in filtered_labels])
    
    return filtered_images, remapped_labels


def load_torch_dataset(dataset_dir):
    """
    Load PyTorch dataset from saved files.
    
    Args:
        dataset_dir: directory containing train.pt and test.pt
        
    Returns:
        train_data: dict with 'images', 'labels', 'label_names'
        test_data: dict with 'images', 'labels', 'label_names'
    """
    dataset_dir = Path(dataset_dir)
    
    # Load training data
    train_data = torch.load(dataset_dir / 'train.pt', weights_only=False)
    
    # Load test data
    test_data = torch.load(dataset_dir / 'test.pt', weights_only=False)
    
    return train_data, test_data


def load_all_cifar10_as_torch(data_dir, normalize=True):
    """
    Load all CIFAR-10 data (not filtered to binary) as PyTorch tensors.
    
    Args:
        data_dir: Path to cifar-10-batches-py directory
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        train_images: torch.Tensor of shape (50000, 3, 32, 32)
        train_labels: torch.Tensor of shape (50000,)
        test_images: torch.Tensor of shape (10000, 3, 32, 32)
        test_labels: torch.Tensor of shape (10000,)
        label_names: list of 10 class names
    """
    # Load all data using existing function
    train_images, train_labels, test_images, test_labels, label_names = load_all_cifar10(data_dir)
    
    # Convert from numpy to torch tensors and from HWC to CHW format
    train_images_tensor = torch.from_numpy(train_images)  # Shape: (50000, 32, 32, 3) HWC
    train_images_tensor = train_images_tensor.permute(0, 3, 1, 2)  # (50000, 3, 32, 32) CHW
    train_labels_tensor = torch.from_numpy(train_labels)
    
    test_images_tensor = torch.from_numpy(test_images)  # Shape: (10000, 32, 32, 3) HWC
    test_images_tensor = test_images_tensor.permute(0, 3, 1, 2)  # (10000, 3, 32, 32) CHW
    test_labels_tensor = torch.from_numpy(test_labels)
    
    if normalize:
        train_images_tensor = train_images_tensor.float() / 255.0
        test_images_tensor = test_images_tensor.float() / 255.0
    
    return train_images_tensor, train_labels_tensor, test_images_tensor, test_labels_tensor, label_names


def save_torch_dataset(train_images, train_labels, test_images, test_labels, 
                       label_names, output_dir):
    """
    Save filtered dataset in PyTorch format.
    
    Args:
        train_images: numpy array of training images
        train_labels: numpy array of training labels
        test_images: numpy array of test images
        test_labels: numpy array of test labels
        label_names: list of class names
        output_dir: directory to save the data
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to PyTorch tensors
    # Keep images as uint8 (0-255) to save space, convert to CHW format
    train_images_tensor = torch.from_numpy(train_images).permute(0, 3, 1, 2).byte()
    train_labels_tensor = torch.from_numpy(train_labels).long()
    
    test_images_tensor = torch.from_numpy(test_images).permute(0, 3, 1, 2).byte()
    test_labels_tensor = torch.from_numpy(test_labels).long()
    
    # Save training data
    torch.save({
        'images': train_images_tensor,
        'labels': train_labels_tensor,
        'label_names': label_names
    }, output_dir / 'train.pt')
    
    # Save test data
    torch.save({
        'images': test_images_tensor,
        'labels': test_labels_tensor,
        'label_names': label_names
    }, output_dir / 'test.pt')
    
    # Save metadata
    metadata = {
        'num_train_samples': len(train_labels),
        'num_test_samples': len(test_labels),
        'num_classes': len(label_names),
        'label_names': label_names,
        'image_shape': list(train_images_tensor.shape[1:]),  # (C, H, W)
        'data_format': 'CHW (Channel, Height, Width)',
        'pixel_range': '[0, 255] (uint8)',
        'dtype': 'uint8 for images, long for labels'
    }
    
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ PyTorch dataset saved to '{output_dir}'")
    print(f"  - train.pt: {len(train_labels)} samples")
    print(f"  - test.pt: {len(test_labels)} samples")
    print(f"  - metadata.json")


def main():
    """
    Main function to load CIFAR-10 dataset, filter for binary classification, 
    and save in PyTorch format.
    """
    # Get data directory
    home_dir = os.path.expanduser('~')
    data_dir = os.path.join(home_dir, 'cifar-10-batches-py')
    
    print(f"Loading CIFAR-10 dataset from: {data_dir}")
    print("=" * 60)
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist!")
        return
    
    # Load all data
    train_images, train_labels, test_images, test_labels, label_names = load_all_cifar10(data_dir)
    
    # Print original dataset statistics
    print(f"\nOriginal dataset loaded successfully!")
    print("=" * 60)
    print(f"Training set: {train_images.shape[0]} images")
    print(f"Test set: {test_images.shape[0]} images")
    print(f"All class names: {label_names}")
    
    # Filter to keep only labels 0 and 1 (airplane and automobile)
    keep_labels = [0, 1]
    binary_label_names = [label_names[i] for i in keep_labels]
    
    print(f"\nFiltering dataset to keep only labels {keep_labels}: {binary_label_names}")
    print("=" * 60)
    
    # Filter training set
    train_images_filtered, train_labels_filtered = filter_binary_classes(
        train_images, train_labels, keep_labels=keep_labels
    )
    
    # Filter test set
    test_images_filtered, test_labels_filtered = filter_binary_classes(
        test_images, test_labels, keep_labels=keep_labels
    )
    
    # Print filtered dataset statistics
    print(f"\nFiltered dataset statistics:")
    print("=" * 60)
    print(f"Training set: {len(train_labels_filtered)} images")
    print(f"  - Images shape: {train_images_filtered.shape}")
    print(f"  - Labels shape: {train_labels_filtered.shape}")
    print(f"\nTest set: {len(test_labels_filtered)} images")
    print(f"  - Images shape: {test_images_filtered.shape}")
    print(f"  - Labels shape: {test_labels_filtered.shape}")
    
    # Print class distribution for filtered data
    print(f"\nBinary class mapping:")
    for new_label, old_label in enumerate(keep_labels):
        print(f"  {new_label} -> {label_names[old_label]} (original label {old_label})")
    
    print(f"\nFiltered training set distribution:")
    unique, counts = np.unique(train_labels_filtered, return_counts=True)
    for label_idx, count in zip(unique, counts):
        print(f"  {binary_label_names[label_idx]:12s} (Label {label_idx}): {count} images")
    
    print(f"\nFiltered test set distribution:")
    unique, counts = np.unique(test_labels_filtered, return_counts=True)
    for label_idx, count in zip(unique, counts):
        print(f"  {binary_label_names[label_idx]:12s} (Label {label_idx}): {count} images")
    
    # Save in PyTorch format (in repository folder)
    output_dir = 'cifar10_binary'
    save_torch_dataset(
        train_images_filtered, 
        train_labels_filtered,
        test_images_filtered,
        test_labels_filtered,
        binary_label_names,
        output_dir
    )
    
    # Verify by loading the saved dataset
    print("\n" + "=" * 60)
    print("Verifying saved dataset...")
    print("=" * 60)
    
    train_data, test_data = load_torch_dataset(output_dir)
    
    print(f"\nLoaded training data:")
    print(f"  - Images tensor shape: {train_data['images'].shape}")
    print(f"  - Images dtype: {train_data['images'].dtype}")
    print(f"  - Images min/max: [{train_data['images'].min()}, {train_data['images'].max()}]")
    print(f"  - Labels tensor shape: {train_data['labels'].shape}")
    print(f"  - Labels dtype: {train_data['labels'].dtype}")
    print(f"  - Label names: {train_data['label_names']}")
    
    print(f"\nLoaded test data:")
    print(f"  - Images tensor shape: {test_data['images'].shape}")
    print(f"  - Images dtype: {test_data['images'].dtype}")
    print(f"  - Images min/max: [{test_data['images'].min()}, {test_data['images'].max()}]")
    print(f"  - Labels tensor shape: {test_data['labels'].shape}")
    print(f"  - Labels dtype: {test_data['labels'].dtype}")
    
    # Show first few datapoints
    print("\n" + "=" * 60)
    print("First 10 training samples:")
    print("=" * 60)
    for i in range(10):
        label_idx = train_data['labels'][i].item()
        label_name = train_data['label_names'][label_idx]
        img_shape = train_data['images'][i].shape
        img_min = train_data['images'][i].min().item()
        img_max = train_data['images'][i].max().item()
        print(f"Sample {i}: Label={label_idx} ({label_name}), "
              f"Shape={img_shape}, Pixel range=[{img_min}, {img_max}]")
    
    print("\n" + "=" * 60)
    print("First 10 test samples:")
    print("=" * 60)
    for i in range(10):
        label_idx = test_data['labels'][i].item()
        label_name = test_data['label_names'][label_idx]
        img_shape = test_data['images'][i].shape
        img_min = test_data['images'][i].min().item()
        img_max = test_data['images'][i].max().item()
        print(f"Sample {i}: Label={label_idx} ({label_name}), "
              f"Shape={img_shape}, Pixel range=[{img_min}, {img_max}]")
    
    return train_images_filtered, train_labels_filtered, test_images_filtered, test_labels_filtered, binary_label_names


if __name__ == "__main__":
    main()
