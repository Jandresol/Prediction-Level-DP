import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.attacks.membership_inference import lira_attack, label_only_attack
from src.attacks.canary_mi import lira_canary_attack, label_only_canary_attack
from src.dpsgd.train_dp_sgd import train_dp_sgd
from src.baseline.train_baseline import train_baseline_cifar10

if __name__ == "__main__":
    lira_attack(number_of_models=100, training_func=train_baseline_cifar10, target_fpr=0.1)
    # lira_canary_attack(number_of_models=100, training_func=train_baseline_cifar10, target_fpr=0.01)
    # label_only_canary_attack(number_of_models=100, training_func=train_baseline_cifar10, target_fpr=0.01)
    # label_only_canary_attack(number_of_models=100, training_func=train_dp_sgd, target_fpr=0.1)