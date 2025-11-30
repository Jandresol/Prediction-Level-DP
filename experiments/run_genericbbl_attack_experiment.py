import sys
import os

# Add project root to path to allow relative imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.attacks.membership_inference import lira_attack
from src.genericbbl.predict_genericbbl import train_genericbbl

if __name__ == "__main__":
    # Run LIRA attack on the GenericBBL training method
    lira_attack(
        number_of_models=100,
        training_func=train_genericbbl,
        target_fpr=0.1
    )
