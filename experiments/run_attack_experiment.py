import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.attacks.membership_inference import membership_inference_attack
from src.dpsgd.train_dp_sgd import train_dp_sgd
from src.baseline.train_baseline import train_baseline_cifar10

if __name__ == "__main__":
    membership_inference_attack(number_of_models=100, training_func=train_baseline_cifar10, target_fpr=0.1)
    membership_inference_attack(number_of_models=1000, training_func=train_baseline_cifar10, target_fpr=0.01)
    membership_inference_attack(number_of_models=1000, training_func=train_dp_sgd, target_fpr=0.01)