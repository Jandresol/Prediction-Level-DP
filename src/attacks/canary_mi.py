"""
Canary-Based Membership Inference Attack

This module performs membership inference attacks focusing on a specific set of
"canary" samples (the most vulnerable samples identified from previous experiments).

Unlike the standard membership inference which evaluates each sample individually,
this approach:
1. Focuses only on the canary samples
2. Evaluates TPR/FPR for the canary set as a whole
3. Trains models on random half of canaries + random half of remaining data
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from src.datasets.load_cifar10 import load_torch_dataset
from src.attacks.membership_inference import phi, get_subset, generate_augmentations_batch


def create_canary_masks(number_of_models, canary_ids, non_canary_ids, num_data_points, device="cuda"):
    """
    Create membership masks for canary-based attacks.
    Each model is trained on random half of canaries + random half of non-canaries.
    Models are created in pairs where model 2k+1 uses the complement of model 2k.
    
    Args:
        number_of_models: Number of models to create masks for
        canary_ids: List of canary sample IDs
        non_canary_ids: List of non-canary sample IDs
        num_data_points: Total number of data points
        device: Device to store masks on
        
    Returns:
        masks: Tensor of shape (number_of_models, num_data_points)
    """
    masks = torch.zeros(number_of_models, num_data_points, device=device)
    
    num_canaries = len(canary_ids)
    num_non_canaries = len(non_canary_ids)
    
    for k in range(number_of_models // 2):
        # Random half of canaries
        num_canaries_in = num_canaries // 2
        selected_canaries = np.random.choice(canary_ids, size=num_canaries_in, replace=False)
        complement_canaries = np.setdiff1d(canary_ids, selected_canaries)
        
        # Random half of non-canaries
        num_non_canaries_in = num_non_canaries // 2
        selected_non_canaries = np.random.choice(non_canary_ids, size=num_non_canaries_in, replace=False)
        complement_non_canaries = np.setdiff1d(non_canary_ids, selected_non_canaries)
        
        # Model 2k: selected canaries + selected non-canaries
        all_selected = np.concatenate([selected_canaries, selected_non_canaries])
        masks[2 * k, all_selected] = 1
        
        # Model 2k+1: complement canaries + complement non-canaries
        all_complement = np.concatenate([complement_canaries, complement_non_canaries])
        masks[2 * k + 1, all_complement] = 1
    
    return masks


def lira_canary_attack(
    number_of_models,
    training_func,
    canary_file="results/vulnerable_samples/top100_baseline_intersection.txt",
    target_fpr=0.1,
    output_prefix="canary_lira"
):
    """
    LIRA-style attack focusing on canary samples.
    
    Args:
        number_of_models: Number of models to train
        training_func: Training function (e.g., train_dp_sgd, train_baseline_cifar10)
        canary_file: Path to text file containing canary sample IDs
        target_fpr: Target false positive rate
        output_prefix: Prefix for output files
    """
    print("=" * 80)
    print("LIRA Canary Attack")
    print("=" * 80)
    
    # Load canary IDs
    print(f"\nLoading canary IDs from: {canary_file}")
    with open(canary_file, 'r') as f:
        canary_ids = [int(line.strip()) for line in f if line.strip()][:50]
    
    num_canaries = len(canary_ids)
    print(f"Number of canaries: {num_canaries}")
    
    # Load dataset
    train_data, test_data = load_torch_dataset("cifar10_binary")
    num_data_points = len(train_data["images"])
    
    # Split into canaries and non-canaries
    canary_ids_set = set(canary_ids)
    non_canary_ids = [i for i in range(num_data_points) if i not in canary_ids_set]
    
    print(f"Total training samples: {num_data_points}")
    print(f"Non-canary samples: {len(non_canary_ids)}")
    
    # Create membership masks with complementary pairs
    masks = create_canary_masks(number_of_models, canary_ids, non_canary_ids, num_data_points, device="cuda")
    
    num_canaries_in = num_canaries // 2
    num_non_canaries_in = len(non_canary_ids) // 2
    
    print(f"\nTraining {number_of_models} models (in complementary pairs)...")
    print(f"Each model trained on ~{num_canaries_in + num_non_canaries_in} samples")
    
    # Compute confidence scores for canaries only
    loss = nn.BCELoss()
    confs = torch.zeros(number_of_models, num_canaries, device="cuda")
    xs = train_data["images"].float().cuda() / 255.0
    ys = train_data["labels"].float().cuda().view(-1, 1)
    batch_size = 128
    
    for k in tqdm(range(number_of_models), desc="Training models and computing confidences"):
        # Train model
        indices = torch.where(masks[k] == 1)[0].cpu()
        train_data_subset = get_subset(train_data, indices=indices)
        model = training_func(train_data_subset, test_data, eval=False)[0]
        model.eval()
        
        # Compute confidence scores for canaries only
        with torch.no_grad():
            for i, canary_id in enumerate(canary_ids):
                x = xs[canary_id:canary_id+1]
                y = ys[canary_id:canary_id+1]
                conf = phi(loss(model(x), y))
                confs[k, i] = conf
        
        # Release model memory
        del model
        torch.cuda.empty_cache()
    
    # For each canary, train logistic regression with cross-validation
    print("\nComputing attack confidence scores for canaries...")
    
    all_confidences = []  # Will store all (in/out, confidence) pairs
    half_models = number_of_models // 2
    
    for i in tqdm(range(num_canaries), desc="Computing LR confidences"):
        canary_id = canary_ids[i]
        
        # Get features for this canary across all models
        features = confs[:, i].unsqueeze(1).cpu().numpy()  # (num_models, 1)
        
        # Get membership labels
        membership = masks[:, canary_id].cpu().numpy()  # (num_models,)
        
        # Cross-validation: split models into two halves
        first_half_indices = np.arange(0, half_models)
        second_half_indices = np.arange(half_models, number_of_models)
        
        # Train lr_model_1 on first half
        lr_model_1 = LogisticRegression(max_iter=100, tol=1e-2, random_state=42)
        lr_model_1.fit(features[first_half_indices], membership[first_half_indices])
        
        # Train lr_model_2 on second half
        lr_model_2 = LogisticRegression(max_iter=100, tol=1e-2, random_state=42)
        lr_model_2.fit(features[second_half_indices], membership[second_half_indices])
        
        # Get confidence scores using cross-validation
        confidences_first_half = lr_model_2.predict_proba(features[first_half_indices])[:, 1]
        confidences_second_half = lr_model_1.predict_proba(features[second_half_indices])[:, 1]
        
        # Combine confidences
        confidences = np.concatenate([confidences_first_half, confidences_second_half])
        
        # Store all (membership, confidence) pairs
        for mem, conf in zip(membership, confidences):
            all_confidences.append((int(mem), float(conf)))
    
    # Convert to arrays
    all_memberships = np.array([x[0] for x in all_confidences])
    all_confs = np.array([x[1] for x in all_confidences])
    
    # Split into in/out
    in_confidences = all_confs[all_memberships == 1]
    out_confidences = all_confs[all_memberships == 0]
    
    print(f"\nTotal confidence scores collected:")
    print(f"  In-training: {len(in_confidences)}")
    print(f"  Out-of-training: {len(out_confidences)}")
    
    # Compute TPR at target FPR
    sorted_out = np.sort(out_confidences)[::-1]  # Descending order
    thresh_idx = min(int(target_fpr * len(out_confidences)), len(sorted_out) - 1)
    threshold = sorted_out[thresh_idx]
    
    tpr = np.mean(in_confidences > threshold)
    
    print(f"\nResults at FPR={target_fpr}:")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  TPR: {tpr:.4f}")
    
    # Compute full ROC curve data
    thresholds = np.sort(np.concatenate([in_confidences, out_confidences]))[::-1]
    fprs_list = []
    tprs_list = []
    
    for thresh in thresholds[::max(1, len(thresholds)//1000)]:
        fpr = np.mean(out_confidences >= thresh)
        tpr_val = np.mean(in_confidences >= thresh)
        fprs_list.append(float(fpr))
        tprs_list.append(float(tpr_val))
    
    # Prepare results with ROC curve data (caller will save with unique filename)
    results = {
        "attack_type": "LIRA",
        "num_canaries": num_canaries,
        "num_models": number_of_models,
        "target_fpr": target_fpr,
        "threshold": float(threshold),
        "tpr": float(tpr),
        "num_in_samples": len(in_confidences),
        "num_out_samples": len(out_confidences),
        "training_function": training_func.__name__,
        "roc_curve": {
            "fprs": fprs_list,
            "tprs": tprs_list
        },
        "confidences": {
            "in": in_confidences.tolist(),
            "out": out_confidences.tolist()
        }
    }
    
    # Create visualization (save to results for backward compatibility)
    vis_dir = Path("results/canary_attacks")
    vis_dir.mkdir(parents=True, exist_ok=True)
    create_roc_plot(in_confidences, out_confidences, target_fpr, tpr, 
                   vis_dir / f"{output_prefix}_{training_func.__name__}_{number_of_models}models_fpr{target_fpr:.2f}.png",
                   f"LIRA Canary Attack - {training_func.__name__}")
    
    return tpr, results


def label_only_canary_attack(
    number_of_models,
    training_func,
    canary_file="results/vulnerable_samples/top100_baseline_intersection.txt",
    target_fpr=0.1,
    output_prefix="canary_label_only"
):
    """
    Label-only attack focusing on canary samples.
    
    Args:
        number_of_models: Number of models to train
        training_func: Training function
        canary_file: Path to text file containing canary sample IDs
        target_fpr: Target false positive rate
        output_prefix: Prefix for output files
    """
    print("=" * 80)
    print("Label-Only Canary Attack")
    print("=" * 80)
    
    # Load canary IDs
    print(f"\nLoading canary IDs from: {canary_file}")
    with open(canary_file, 'r') as f:
        canary_ids = [int(line.strip()) for line in f if line.strip()]
    
    num_canaries = len(canary_ids)
    print(f"Number of canaries: {num_canaries}")
    
    # Load dataset
    train_data, test_data = load_torch_dataset("cifar10_binary")
    num_data_points = len(train_data["images"])
    
    # Split into canaries and non-canaries
    canary_ids_set = set(canary_ids)
    non_canary_ids = [i for i in range(num_data_points) if i not in canary_ids_set]
    
    print(f"Total training samples: {num_data_points}")
    print(f"Non-canary samples: {len(non_canary_ids)}")
    
    # Create membership masks with complementary pairs
    masks = create_canary_masks(number_of_models, canary_ids, non_canary_ids, num_data_points, device="cuda")
    
    print(f"\nTraining {number_of_models} models (in complementary pairs)...")
    
    # Compute augmentation predictions for canaries only
    aug_predictions = torch.zeros(number_of_models, num_canaries, 18, device="cuda")
    xs = train_data["images"].float().cuda() / 255.0
    ys = train_data["labels"].long().cuda()
    
    for k in tqdm(range(number_of_models), desc="Training models and computing augmentation predictions"):
        # Train model
        indices = torch.where(masks[k] == 1)[0].cpu()
        train_data_subset = get_subset(train_data, indices=indices)
        model = training_func(train_data_subset, test_data, eval=False)[0]
        model.eval()
        
        # Process canaries in batches
        batch_size = 64
        canary_indices_tensor = torch.tensor(canary_ids, device="cuda")
        
        with torch.no_grad():
            for batch_start in range(0, num_canaries, batch_size):
                batch_end = min(batch_start + batch_size, num_canaries)
                batch_canary_ids = canary_ids[batch_start:batch_end]
                
                batch_xs = xs[batch_canary_ids]
                batch_ys = ys[batch_canary_ids]
                
                # Generate augmentations
                augmented_batch = generate_augmentations_batch(batch_xs)
                
                # Reshape and predict
                batch_size_actual = augmented_batch.shape[0]
                augmented_flat = augmented_batch.reshape(batch_size_actual * 18, 3, 32, 32)
                
                outputs = model(augmented_flat)
                preds = (outputs > 0.5).long().squeeze()
                preds = preds.reshape(batch_size_actual, 18)
                
                # Check correctness
                batch_ys_expanded = batch_ys.squeeze().unsqueeze(1).expand(-1, 18)
                correct = (preds == batch_ys_expanded).float()
                
                aug_predictions[k, batch_start:batch_end] = correct
        
        # Release model memory
        del model
        torch.cuda.empty_cache()
    
    # For each canary, train logistic regression with cross-validation
    print("\nComputing attack confidence scores for canaries...")
    
    all_confidences = []
    half_models = number_of_models // 2
    
    for i in tqdm(range(num_canaries), desc="Computing LR confidences"):
        canary_id = canary_ids[i]
        
        # Get 18-bit features for this canary across all models
        features = aug_predictions[:, i, :].cpu().numpy()  # (num_models, 18)
        
        # Get membership labels
        membership = masks[:, canary_id].cpu().numpy()
        
        # Cross-validation
        first_half_indices = np.arange(0, half_models)
        second_half_indices = np.arange(half_models, number_of_models)
        
        # Train two LR models
        lr_model_1 = LogisticRegression(max_iter=100, tol=1e-2, random_state=42)
        lr_model_1.fit(features[first_half_indices], membership[first_half_indices])
        
        lr_model_2 = LogisticRegression(max_iter=100, tol=1e-2, random_state=42)
        lr_model_2.fit(features[second_half_indices], membership[second_half_indices])
        
        # Get confidence scores
        confidences_first_half = lr_model_2.predict_proba(features[first_half_indices])[:, 1]
        confidences_second_half = lr_model_1.predict_proba(features[second_half_indices])[:, 1]
        
        confidences = np.concatenate([confidences_first_half, confidences_second_half])
        
        # Store all pairs
        for mem, conf in zip(membership, confidences):
            all_confidences.append((int(mem), float(conf)))
    
    # Convert to arrays
    all_memberships = np.array([x[0] for x in all_confidences])
    all_confs = np.array([x[1] for x in all_confidences])
    
    in_confidences = all_confs[all_memberships == 1]
    out_confidences = all_confs[all_memberships == 0]
    
    print(f"\nTotal confidence scores collected:")
    print(f"  In-training: {len(in_confidences)}")
    print(f"  Out-of-training: {len(out_confidences)}")
    
    # Compute TPR at target FPR
    sorted_out = np.sort(out_confidences)[::-1]
    thresh_idx = min(int(target_fpr * len(out_confidences)), len(sorted_out) - 1)
    threshold = sorted_out[thresh_idx]
    
    tpr = np.mean(in_confidences > threshold)
    
    print(f"\nResults at FPR={target_fpr}:")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  TPR: {tpr:.4f}")
    
    # Compute full ROC curve data
    thresholds = np.sort(np.concatenate([in_confidences, out_confidences]))[::-1]
    fprs_list = []
    tprs_list = []
    
    for thresh in thresholds[::max(1, len(thresholds)//1000)]:
        fpr = np.mean(out_confidences >= thresh)
        tpr_val = np.mean(in_confidences >= thresh)
        fprs_list.append(float(fpr))
        tprs_list.append(float(tpr_val))
    
    # Prepare results with ROC curve data (caller will save with unique filename)
    results = {
        "attack_type": "Label-Only",
        "num_canaries": num_canaries,
        "num_models": number_of_models,
        "target_fpr": target_fpr,
        "threshold": float(threshold),
        "tpr": float(tpr),
        "num_in_samples": len(in_confidences),
        "num_out_samples": len(out_confidences),
        "training_function": training_func.__name__,
        "roc_curve": {
            "fprs": fprs_list,
            "tprs": tprs_list
        },
        "confidences": {
            "in": in_confidences.tolist(),
            "out": out_confidences.tolist()
        }
    }
    
    # Create visualization (save to results for backward compatibility)
    vis_dir = Path("results/canary_attacks")
    vis_dir.mkdir(parents=True, exist_ok=True)
    create_roc_plot(in_confidences, out_confidences, target_fpr, tpr,
                   vis_dir / f"{output_prefix}_{training_func.__name__}_{number_of_models}models_fpr{target_fpr:.2f}.png",
                   f"Label-Only Canary Attack - {training_func.__name__}")
    
    return tpr, results


def create_roc_plot(in_confidences, out_confidences, target_fpr, tpr_at_target, output_file, title):
    """
    Create ROC curve and confidence distribution plots.
    
    Args:
        in_confidences: Confidence scores for in-training samples
        out_confidences: Confidence scores for out-of-training samples
        target_fpr: Target FPR
        tpr_at_target: TPR at target FPR
        output_file: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Confidence distributions
    ax = axes[0]
    ax.hist(out_confidences, bins=50, alpha=0.5, label='Out-of-training', color='blue', density=True)
    ax.hist(in_confidences, bins=50, alpha=0.5, label='In-training', color='red', density=True)
    ax.set_xlabel('Confidence Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Confidence Score Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: ROC curve with log-log scale
    ax = axes[1]
    
    # Compute ROC curve points
    thresholds = np.sort(np.concatenate([in_confidences, out_confidences]))[::-1]
    fprs = []
    tprs = []
    
    for thresh in thresholds[::max(1, len(thresholds)//1000)]:  # Sample for efficiency
        fpr = np.mean(out_confidences >= thresh)
        tpr_val = np.mean(in_confidences >= thresh)
        fprs.append(max(fpr, 1e-6))  # Avoid log(0)
        tprs.append(max(tpr_val, 1e-6))  # Avoid log(0)
    
    # Convert to arrays
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    
    ax.plot(fprs, tprs, linewidth=2, label='ROC Curve')
    
    # Random guess line in log-log scale
    log_fprs = np.logspace(-6, 0, 100)
    ax.plot(log_fprs, log_fprs, 'k--', alpha=0.3, label='Random Guess')
    
    # Operating point
    ax.scatter([max(target_fpr, 1e-6)], [max(tpr_at_target, 1e-6)], color='red', s=100, zorder=5, 
               label=f'Operating Point\n(FPR={target_fpr:.2f}, TPR={tpr_at_target:.2f})')
    
    ax.set_xlabel('False Positive Rate (log scale)', fontsize=11)
    ax.set_ylabel('True Positive Rate (log scale)', fontsize=11)
    ax.set_title('ROC Curve (Log-Log Scale)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([1e-3, 1])
    ax.set_ylim([1e-3, 1])
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {output_file}")


if __name__ == "__main__":
    from src.baseline.train_baseline import train_baseline_cifar10
    
    # Example usage
    print("Running LIRA canary attack on baseline model...")
    lira_canary_attack(100, train_baseline_cifar10, target_fpr=0.1)
    
    print("\n" + "=" * 80 + "\n")
    
    print("Running Label-Only canary attack on baseline model...")
    label_only_canary_attack(100, train_baseline_cifar10, target_fpr=0.1)

