import sys
import os

# Add project root to path to allow relative imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.attacks.membership_inference import lira_attack
from src.attacks.canary_mi import lira_canary_attack, label_only_canary_attack
from src.genericbbl.predict_genericbbl import train_genericbbl

if __name__ == "__main__":
    # lira_attack(number_of_models=100, training_func=train_genericbbl, target_fpr=0.1)
    label_only_canary_attack(number_of_models=100, training_func=train_genericbbl, target_fpr=0.01)
