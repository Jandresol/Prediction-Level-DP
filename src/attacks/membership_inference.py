import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from sklearn.linear_model import LogisticRegression


from src.datasets.load_cifar10 import load_torch_dataset
from src.dpsgd.train_dp_sgd import train_dp_sgd


def phi(x):
    x = torch.exp(-x)
    assert x.min() >= 0 and x.max() <= 1, f"x must be between 0 and 1, but got min={x.min()} and max={x.max()}"
    return torch.log(x / (1 - x))


def get_subset(data, indices):
    return {
        "images": data["images"][indices],
        "labels": data["labels"][indices],
        "label_names": data["label_names"]
    }

    
def lira_attack(number_of_models, training_func, target_fpr=0.1):
    """
    LIRA-style membership inference attack using logistic regression.
    
    Train a number of models on the dataset in complementary pairs, each model's training 
    set consists of a random half of the data.
    
    For each data point:
    1. Extract loss-based confidence scores across all models
    2. Train logistic regression with cross-validation to predict membership
    3. Find threshold where FPR = target_fpr and compute corresponding TPR
    
    Finally draw histograms for the TPR gathered from last step.
    """
    train_data, test_data = load_torch_dataset("cifar10_binary")
    
    num_data_points = len(train_data["images"])
    masks = torch.zeros(number_of_models, num_data_points, device="cuda")
    for k in range(number_of_models//2):
        sample_indices = np.random.choice(num_data_points, size=num_data_points//2, replace=False)
        reversed_sample_indices = np.setdiff1d(np.arange(num_data_points), sample_indices)
        masks[2 * k, sample_indices] = 1
        masks[2 * k + 1, reversed_sample_indices] = 1
        
    loss = nn.BCELoss()
    confs = torch.zeros(number_of_models, num_data_points, device="cuda")
    xs = train_data["images"].float().cuda() / 255.0
    ys = train_data["labels"].float().cuda().view(-1, 1)
    batch_size = 128
    
    for k in tqdm(range(number_of_models), desc="Training models and computing confidences"):
        # Train model
        indices = torch.where(masks[k] == 1)[0].cpu()
        train_data_subset = get_subset(train_data, indices=indices)
        model = training_func(train_data_subset, test_data, eval=False)[0]
        model.eval()
        
        # Compute confidence scores
        for i in range(0, num_data_points, batch_size):
            x = xs[i:i+batch_size]
            y = ys[i:i+batch_size]
            with torch.no_grad():
                conf = phi(loss(model(x), y))
                confs[k, i:i+batch_size] = conf
        
        # Release model memory
        del model
        torch.cuda.empty_cache()
                
    # For each data point, train logistic regression with cross-validation to predict membership
    tprs = []
    half_models = number_of_models // 2
    
    for i in tqdm(range(num_data_points), desc="Computing TPRs via logistic regression"):
        # Get features for this data point across all models
        features = confs[:, i].unsqueeze(1).cpu().numpy()  # (num_models, 1)
        
        # Get membership labels (in training set or not)
        membership = masks[:, i].cpu().numpy()  # (num_models,)
        
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
        # First half predictions come from model trained on second half
        confidences_first_half = lr_model_2.predict_proba(features[first_half_indices])[:, 1]
        # Second half predictions come from model trained on first half
        confidences_second_half = lr_model_1.predict_proba(features[second_half_indices])[:, 1]
        
        # Combine confidences
        confidences = np.concatenate([confidences_first_half, confidences_second_half])
        
        # Split into in/out samples
        in_confidences = confidences[membership == 1]
        out_confidences = confidences[membership == 0]
        
        # Find threshold where FPR = target_fpr
        if len(out_confidences) > 0:
            sorted_out = np.sort(out_confidences)[::-1]  # Descending order
            thresh_idx = min(int(target_fpr * len(out_confidences)), len(sorted_out) - 1)
            threshold = sorted_out[thresh_idx]
            
            # Compute TPR at this threshold
            tpr = np.mean(in_confidences > threshold) if len(in_confidences) > 0 else 0.0
            tprs.append(tpr)
        else:
            tprs.append(0.0)
        
        
    # Save the TPRs to a json file
    with open(f"results/{training_func.__name__}_lira_tprs_{number_of_models}models_fpr{target_fpr:.2f}.json", "w") as f:
        json.dump(tprs, f)
        
        
    # Plot a diagram of the sorted TPRs
    plt.plot(sorted(tprs))
    plt.xlabel("Data point index")
    plt.ylabel("TPR")
    plt.title("Membership inference attack")
    plt.ylim(0, 1)
    plt.xlim(0, len(tprs))
    
    # Use the function name to save the figure
    plt.savefig(f"results/{training_func.__name__}_lira_tprs_{number_of_models}models_fpr{target_fpr:.2f}.png")
    plt.close()
        

def generate_augmentations_batch(images):
    """
    Generate 18 augmentations for a batch of images
    Translations: [-4, 0, +4] pixels in x and y directions
    
    Args:
        images: tensor of shape (B, C, H, W)
    
    Returns:
        augmented: tensor of shape (B, 18, C, H, W)
    """
    batch_size = images.shape[0]
    augmentations = []
    
    # 9 translations
    for dx in [-4, 0, 4]:
        for dy in [-4, 0, 4]:
            # Pad and crop to translate
            if dx == 0 and dy == 0:
                translated = images
            else:
                # Pad images
                padded = F.pad(images, (4, 4, 4, 4), mode='constant', value=0)
                # Crop to translate
                translated = padded[:, :, 4+dy:4+dy+32, 4+dx:4+dx+32]
            
            # Original and horizontally flipped
            augmentations.append(translated)
            augmentations.append(torch.flip(translated, dims=[3]))  # Flip along width
    
    # Stack to (18, B, C, H, W) and transpose to (B, 18, C, H, W)
    return torch.stack(augmentations).transpose(0, 1)


def label_only_attack(number_of_models, training_func, target_fpr=0.1):
    """
    Similar to LIRA attack, but we don't get to compute the loss on the data. Instead, after training each model on the partial data, for each data point, we
    1. generate a total of 18 augmentations ([-4, +4] pixels in each direction * horizontal flip).
    2. feed the augmentations to the model and get a 18 bit binary vector of whether the model predicted the correct label for each augmentation;
    3. let the 18 bit vector be input and the membership bit be output, train a logistic regression model to predict the membership bit from the 18 bit vector;
    4. for each data point, find a confidence threshold where FPR is target_fpr and get the corresponding TPR;
    5. draw a histogram of the TPRs gathered from last step and output data and figure like above.
    """
    train_data, test_data = load_torch_dataset("cifar10_binary")
    
    num_data_points = len(train_data["images"])
    masks = torch.zeros(number_of_models, num_data_points, device="cuda")
    for k in range(number_of_models//2):
        sample_indices = np.random.choice(num_data_points, size=num_data_points//2, replace=False)
        reversed_sample_indices = np.setdiff1d(np.arange(num_data_points), sample_indices)
        masks[2 * k, sample_indices] = 1
        masks[2 * k + 1, reversed_sample_indices] = 1
    
    # Store augmentation prediction results: (num_models, num_data_points, 18)
    aug_predictions = torch.zeros(number_of_models, num_data_points, 18, device="cuda")
    xs = train_data["images"].float().cuda() / 255.0
    ys = train_data["labels"].long().cuda()
    
    for k in tqdm(range(number_of_models), desc="Training models and computing augmentation predictions"):
        # Train model
        indices = torch.where(masks[k] == 1)[0].cpu()
        train_data_subset = get_subset(train_data, indices=indices)
        model = training_func(train_data_subset, test_data, eval=False)[0]
        model.eval()
        
        # Process data points in batches
        batch_size = 64  # Process 64 data points at once
        with torch.no_grad():
            for batch_start in range(0, num_data_points, batch_size):
                batch_end = min(batch_start + batch_size, num_data_points)
                batch_xs = xs[batch_start:batch_end]  # (batch_size, C, H, W)
                batch_ys = ys[batch_start:batch_end]  # (batch_size, 1)
                
                # Generate 18 augmentations for all images in batch
                augmented_batch = generate_augmentations_batch(batch_xs)  # (batch_size, 18, C, H, W)
                
                # Reshape to (batch_size * 18, C, H, W) for model forward pass
                batch_size_actual = augmented_batch.shape[0]
                augmented_flat = augmented_batch.reshape(batch_size_actual * 18, 3, 32, 32)
                
                # Get model predictions on all augmentations
                outputs = model(augmented_flat)  # (batch_size * 18, 1)
                preds = (outputs > 0.5).long().squeeze()  # (batch_size * 18,)
                
                # Reshape back to (batch_size, 18)
                preds = preds.reshape(batch_size_actual, 18)
                
                # Check if predictions match true labels
                batch_ys_expanded = batch_ys.squeeze().unsqueeze(1).expand(-1, 18)  # (batch_size, 18)
                correct = (preds == batch_ys_expanded).float()  # (batch_size, 18)
                
                # Store results
                aug_predictions[k, batch_start:batch_end] = correct
        
        # Release model memory
        del model
        torch.cuda.empty_cache()
    
    # For each data point, train logistic regression and compute TPR
    # Use cross-validation: split models into 2 halves, train on one half and predict on the other
    tprs = []
    half_models = number_of_models // 2
    
    for i in tqdm(range(num_data_points), desc="Computing TPRs via logistic regression"):
        # Get 18-bit features for all models for this data point
        features = aug_predictions[:, i, :].cpu().numpy()  # (num_models, 18)
        
        # Get membership labels (in training set or not)
        membership = masks[:, i].cpu().numpy()  # (num_models,)
        
        # Split models into two halves
        first_half_indices = np.arange(0, half_models)
        second_half_indices = np.arange(half_models, number_of_models)
        
        # Train lr_model_1 on first half
        lr_model_1 = LogisticRegression(max_iter=100, tol=1e-2, random_state=42)
        lr_model_1.fit(features[first_half_indices], membership[first_half_indices])
        
        # Train lr_model_2 on second half
        lr_model_2 = LogisticRegression(max_iter=100, tol=1e-2, random_state=42)
        lr_model_2.fit(features[second_half_indices], membership[second_half_indices])
        
        # Get confidence scores using cross-validation
        # First half predictions come from model trained on second half
        confidences_first_half = lr_model_2.predict_proba(features[first_half_indices])[:, 1]
        # Second half predictions come from model trained on first half
        confidences_second_half = lr_model_1.predict_proba(features[second_half_indices])[:, 1]
        
        # Combine confidences
        confidences = np.concatenate([confidences_first_half, confidences_second_half])
        
        # Split into in/out samples
        in_confidences = confidences[membership == 1]
        out_confidences = confidences[membership == 0]
        
        # Find threshold where FPR = target_fpr
        if len(out_confidences) > 0:
            sorted_out = np.sort(out_confidences)[::-1]  # Descending order
            thresh_idx = min(int(target_fpr * len(out_confidences)), len(sorted_out) - 1)
            threshold = sorted_out[thresh_idx]
            
            # Compute TPR at this threshold
            tpr = np.mean(in_confidences > threshold) if len(in_confidences) > 0 else 0.0
            tprs.append(tpr)
        else:
            tprs.append(0.0)
    
    # Save the TPRs to a json file
    with open(f"results/{training_func.__name__}_label_only_tprs_{number_of_models}models_fpr{target_fpr:.2f}.json", "w") as f:
        json.dump(tprs, f)
    
    # Plot a diagram of the sorted TPRs
    plt.figure(figsize=(10, 6))
    plt.plot(sorted(tprs))
    plt.xlabel("Data point index (sorted)")
    plt.ylabel("TPR")
    plt.title(f"Label-Only Membership Inference Attack (FPR={target_fpr})")
    plt.ylim(0, 1)
    plt.xlim(0, len(tprs))
    plt.grid(True, alpha=0.3)
    
    # Use the function name to save the figure
    plt.savefig(f"results/{training_func.__name__}_label_only_tprs_{number_of_models}models_fpr{target_fpr:.2f}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nLabel-Only Attack Results:")
    print(f"  Mean TPR: {np.mean(tprs):.4f}")
    print(f"  Median TPR: {np.median(tprs):.4f}")
    print(f"  Max TPR: {np.max(tprs):.4f}")
    print(f"  Min TPR: {np.min(tprs):.4f}")
                
                
if __name__ == "__main__":
    lira_attack(100, train_dp_sgd)
    
    