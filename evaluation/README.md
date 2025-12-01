# Evaluation Pipeline

This directory contains scripts for running membership inference attacks and visualizing results.

## Directory Structure

```
evaluation/
├── data/           # Attack results (JSON files with ROC curve data)
├── figures/        # Generated ROC curve plots
├── plot_all_roc_curves.py  # Script to generate plots from results
└── README.md       # This file
```

## Workflow

### 1. Run All Attacks

Run membership inference attacks on all algorithm configurations:

```bash
python experiments/run_attack_experiment.py
```

This will:
- Load hyperparameters from `experiments/hyperparams.json`
- Run **LIRA** and **Label-Only** attacks on:
  - **Baseline** model (non-private)
  - **DP-SGD** models (4 configurations with different noise multipliers)
- Run **Label-Only** attacks on:
  - **GenericBBL** models (4 configurations with different epsilon values)
- Save results to `evaluation/data/` as JSON files
- Each result contains:
  - Full ROC curve data (FPRs and TPRs)
  - Confidence scores (in-training and out-of-training)
  - Attack metadata (algorithm, hyperparameters, etc.)

**Expected Runtime**: Several hours (depends on `num_models` parameter, default is 100)

### 2. Generate ROC Curve Plots

After running attacks, generate visualization plots:

```bash
python evaluation/plot_all_roc_curves.py
```

This will create three types of plots in `evaluation/figures/`:

1. **`combined_roc_curves.png`**
   - All ROC curves on one plot
   - Color-coded by algorithm (blue=baseline, red=DP-SGD, green=GenericBBL)
   - Line style indicates attack type (solid=LIRA, dashed=Label-Only)

2. **`roc_curves_<algorithm>.png`** (one per algorithm)
   - Compares different hyperparameter configurations for each algorithm
   - E.g., `roc_curves_dpsgd.png` shows DP-SGD with different noise multipliers

3. **`roc_curves_by_attack_type.png`**
   - Compares LIRA vs Label-Only attacks across all configurations

All plots use log-log scale for better visualization at low FPR values.

## Configuration

### Hyperparameters

Edit `experiments/hyperparams.json` to modify:

- Number of epochs
- Batch size  
- Privacy parameters (noise multiplier for DP-SGD, epsilon for GenericBBL)
- Learning rate

### Attack Parameters

Edit `experiments/run_attack_experiment.py` to modify:

- `num_models`: Number of shadow models to train (default: 100)
  - More models = better attack accuracy but slower
- `target_fpr`: Target false positive rate (default: 0.01)
  - Used for computing TPR at specific operating point

### Canary Samples

Attacks focus on "canary" samples (most vulnerable). Edit:

```python
canary_file="results/vulnerable_samples/top100_baseline_intersection.txt"
```

in canary attack functions to use different canary sets.

## Output Files

### Attack Results (`evaluation/data/`)

Each JSON file contains:
```json
{
  "attack_type": "LIRA" or "Label-Only",
  "algorithm": "baseline", "dpsgd", or "genericbbl",
  "hyperparameters": {...},
  "num_models": 100,
  "target_fpr": 0.01,
  "tpr": 0.XX,
  "roc_curve": {
    "fprs": [...],  // False positive rates
    "tprs": [...]   // True positive rates
  },
  "confidences": {
    "in": [...],    // Confidence scores for in-training samples
    "out": [...]    // Confidence scores for out-of-training samples
  }
}
```

### Summary File

`evaluation/data/all_attacks_summary.json` contains a list of all attack results.

## Example Analysis

After running the pipeline:

1. **Compare privacy mechanisms**: Which provides better protection - DP-SGD or GenericBBL?
2. **Privacy-utility tradeoff**: How does increasing noise/epsilon affect attack success?
3. **Attack effectiveness**: Is LIRA or Label-Only more effective?
4. **Vulnerability analysis**: Which samples (canaries) are most vulnerable?

## Notes

- Attacks are computationally expensive (training 100+ models)
- Consider using fewer models for quick tests
- Results are non-deterministic (random data splits)
- Log-log scale ROC curves emphasize low FPR region (most relevant for privacy)

