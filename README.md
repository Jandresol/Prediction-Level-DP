# Evaluating ML Differential Privacy at the Prediction Level

This repository implements and compares **Differentially Private Stochastic Gradient Descent (DP-SGD)** and **Generic Bounding Box Locus (GenericBBL)** to study privacy–utility trade-offs in machine learning.  
The baseline experiment trains a DP-SGD model on the **UCI Adult** dataset using [Opacus](https://opacus.ai/).

---

## 🧠 Overview

- **Goal:** Empirically compare training-level vs. prediction-level differential privacy.  
- **DP-SGD:** Adds Gaussian noise to gradients during training to ensure parameter-level privacy.  
- **GenericBBL:** Adds calibrated noise to model predictions during inference for output-level privacy.  
- **Metrics:** Privacy cost (ε), model loss/accuracy, and runtime efficiency.

---

## 📂 Project Structure

```
Prediction-Level-DP/
├── experiments/
│ └── run_dpsgd_experiment.py # Main entry point for DP-SGD training
├── src/
│ ├── datasets/
│ │ └── load_adult.py # Loads and preprocesses the UCI Adult dataset
│ ├── dpsgd/
│ │ └── train_dp_sgd.py # Core DP-SGD training logic
│ ├── models/
│ │ └── adult_mlp.py # Simple MLP model for tabular data
│ ├── config/
│ │ └── dpsgd_config.json # Training and DP hyperparameters
│ └── init.py
├── requirements.txt
└── README.md
```

## ⚙️ Setup Instructions

### 1. Clone the repository

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Prediction-Level-DP.git
cd Prediction-Level-DP
```

### 2. Create and activate a virtual environment
``` 
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate.ps1   # Windows PowerShell
```

### 3. Install dependencies
``` 
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the DP-SGD Experiment
```
python -m experiments.run_dpsgd_experiment
```

Results (accuracy, ε, runtime) are printed to the console and optionally saved to:

```
results/metrics/dpsgd_adult.json
```