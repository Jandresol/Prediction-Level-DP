import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


from src.datasets.load_cifar10 import load_torch_dataset
from src.dpsgd.train_dp_sgd import train_dp_sgd


def phi(x):
    x = torch.exp(-x)
    assert x.min() > 0 and x.max() < 1, "x must be between 0 and 1"
    return torch.log(x / (1 - x))


def get_subset(data, indices):
    return {
        "images": data["images"][indices],
        "labels": data["labels"][indices],
        "label_names": data["label_names"]
    }

    
def membership_inference_attack(number_of_models, training_func, target_fpr=0.1):
    """
    Train a number of models on the dataset, each model's training set consists of a random half of the data.
    
    Then for each data point, compute the average confidence of the models trained on and not the data point. Find a threshold where FPR is 0.01 and get the corresponding TPR.
    
    Finally draw a histograms for the TPR gathered from last step.
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
                
    # For each data point, use a gaussian to estimate the distribution of the confidence scores for in and out of the training set.
    tprs = []
    for i in tqdm(range(num_data_points), desc="Computing TPRs"):
        in_confs = confs[:, i][masks[:, i] == 1]
        out_confs = confs[:, i][masks[:, i] == 0]
        in_mean, in_std = torch.mean(in_confs), torch.std(in_confs)
        out_mean, out_std = torch.mean(out_confs), torch.std(out_confs)
        in_dist = torch.distributions.Normal(in_mean, in_std)
        out_dist = torch.distributions.Normal(out_mean, out_std)
        
        # Get density ratio of each confidence score in in-distribution to out-distribution
        in_density = in_dist.log_prob(confs[:, i])
        out_density = out_dist.log_prob(confs[:, i])
        dr = torch.exp(in_density - out_density)
        
        in_dr = dr[masks[:, i] == 1]
        out_dr = dr[masks[:, i] == 0]
        
        thresh = torch.sort(out_dr, descending=True)[0][round(target_fpr * len(out_dr))]
        tpr = torch.mean((in_dr > thresh).float()).cpu().item()
        tprs.append(tpr)
        
        
    # Save the TPRs to a json file
    with open(f"results/{training_func.__name__}_tprs_{number_of_models}models_fpr{target_fpr:.2f}.json", "w") as f:
        json.dump(tprs, f)
        
        
    # Plot a diagram of the sorted TPRs
    plt.plot(sorted(tprs))
    plt.xlabel("Data point index")
    plt.ylabel("TPR")
    plt.title("Membership inference attack")
    plt.ylim(0, 1)
    plt.xlim(0, len(tprs))
    
    # Use the function name to save the figure
    plt.savefig(f"results/{training_func.__name__}_tprs_{number_of_models}models_fpr{target_fpr:.2f}.png")
    plt.close()
        
        
if __name__ == "__main__":
    membership_inference_attack(100, train_dp_sgd)
    
    