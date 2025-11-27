import numpy as np
import torch
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

from .between_thresholds import BetweenThresholds

class PrivateEverlastingPredictor:
    """
    Implements Algorithm GenericBBL for Private Everlasting Prediction.
    Source: arXiv:2305.09579v1
    """

    def __init__(self, base_learner, vc_dim, epsilon=0.9, delta=0.01, alpha=0.1, beta=0.1, practical_mode=True, device=None):
        """
        :param base_learner: A non-private sklearn-compatible classifier (Concept class C).
        :param vc_dim: The VC dimension of the concept class C.
        :param epsilon: Privacy parameter epsilon.
        :param delta: Privacy parameter delta.
        :param alpha: Error parameter.
        :param beta: Failure probability parameter.
        :param practical_mode: If True, scales down theoretical constants to run on standard hardware.
        :param device: The torch device to use for computation (e.g., 'cuda:0' or 'cpu').
        """
        self.base_learner = base_learner
        self.vc_dim = vc_dim
        self.epsilon = epsilon
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.practical_mode = practical_mode
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize round parameters
        self.current_alpha = alpha / 2.0  # alpha_1
        self.current_beta = beta / 2.0    # beta_1
        self.round_counter = 1
        
        # Storage for the current labeled training set S_i
        self.S_current_X = None
        self.S_current_y = None

    def _calculate_params(self, alpha, beta):
        """
        Calculates n, T_i, lambda_i, R_i based on Step 1, 2, 3a, 3c.
        """
        # Tau constant (Theoretical > 1.1e10)
        tau = 1.1e10 if not self.practical_mode else 10.0
        
        # Lambda_i: Samples sufficient for PAC learning
        # Formula: (8*VC*log(13/alpha) + 4*log(2/beta)) / alpha
        lambda_i = (8 * self.vc_dim * np.log(13 / alpha) + 4 * np.log(2 / beta)) / alpha
        lambda_i = int(np.ceil(lambda_i))

        # T_i: Number of disjoint databases/classifiers
        # Simplified theoretical scaling for implementation
        T_numerator = tau * lambda_i * np.log(1/self.delta)
        T_denominator = alpha * self.epsilon
        T_i = int(np.ceil(T_numerator / T_denominator))
        
        if self.practical_mode:
            # Hard clamp for running on a laptop
            T_i = max(5, int(T_i / 1e8)) 
            lambda_i = max(10, int(lambda_i / 10))

        # R_i: Number of queries to answer in this round
        # S_i size is approx T_i * lambda_i
        size_S_i = T_i * lambda_i
        R_i = int((25600 * size_S_i) / self.epsilon)
        
        if self.practical_mode:
            R_i = min(1000, R_i) # Limit queries per round for demo

        return lambda_i, T_i, R_i

    def train_initial(self, X, y):
        """
        Step 1-2: Ingest initial labeled database S.
        """
        # In a strict implementation, we would check if len(X) meets the theoretical requirement 'n'.
        # Convert to tensors and move to the designated device.
        self.S_current_X = torch.from_numpy(np.array(X)).float().to(self.device)
        self.S_current_y = torch.from_numpy(np.array(y)).long().to(self.device)

        print(f"Initial training complete. Dataset size: {self.S_current_X.shape[0]} on device: {self.device}")

    def run_round_without_privacy(self, query_stream):
        """
        Executes one round (i) of GenericBBL without privacy.

        :param query_stream: An iterator/list of unlabeled query points x.
        :return: List of predictions (0 or 1).
        """
        predictions = []

        #Train base learner on current labeled set
        self.base_learner.fit(self.S_current_X.cpu().numpy(), self.S_current_y.cpu().numpy())
        
        for x_query in query_stream:
            x_query = x_query.reshape(1, -1)
            pred = self.base_learner.predict(x_query)[0]
            predictions.append(pred)
            
        return predictions


    def run_round(self, query_stream):
        """
        Executes one round (i) of GenericBBL.
        
        :param query_stream: An iterator/list of unlabeled query points x.
        :return: List of predictions (0 or 1).
        """
        lambda_i, T_i, R_i = self._calculate_params(self.current_alpha, self.current_beta)
        
        print(f"--- Round {self.round_counter} ---")
        print(f"Params: lambda={lambda_i}, T={T_i}, Max Queries R={R_i}")

        # --- Step 3a: Divide S_i into T_i disjoint databases ---
        # We assume S_current is large enough. If not, we resample (bootstrapping) 
        # to simulate the sufficient data availability assumed by the realizable setting.
        current_size = self.S_current_X.shape[0]
        required_size = T_i * lambda_i
        
        if current_size < required_size:
            print(f"Warning: Data size {current_size} < Required {required_size}. Resampling.")
            # More even resampling: Repeat the full dataset and randomly sample the remainder.
            num_repeats = required_size // current_size
            num_remaining = required_size % current_size
            
            # Start with the repeated full dataset
            original_indices = torch.arange(current_size, device=self.device)
            indices = original_indices.repeat(num_repeats)
            
            if num_remaining > 0:
                # Add the randomly sampled remainder
                remaining_indices = torch.randint(0, current_size, (num_remaining,), device=self.device)
                indices = torch.cat([indices, remaining_indices])
            
            # Shuffle the final combined indices to ensure randomness
            shuffle_perm = torch.randperm(indices.shape[0], device=self.device)
            indices = indices[shuffle_perm]
        else:
            # Sample without replacement
            indices = torch.randperm(current_size, device=self.device)[:required_size]
            
        S_pool_X = self.S_current_X[indices]
        S_pool_y = self.S_current_y[indices]

        # --- Step 3b: Train Ensemble F_i ---
        F_i = []
        for t in range(T_i):
            start = t * lambda_i
            end = (t + 1) * lambda_i
            
            clf = clone(self.base_learner)
            # If the learner has a `set_device` method (like our PyTorch wrapper), use it.
            if hasattr(clf, 'set_device'):
                clf.set_device(self.device)

            # Handle single-class edge case in subsample
            sub_y = S_pool_y[start:end]
            if len(torch.unique(sub_y)) < 2:
                 # Fallback to random prediction if subsample is pure (rare in realizable)
                 clf = DummyClassifier(sub_y[0].item())
            else:
                # Convert back to numpy for the scikit-learn compatible fit method
                clf.fit(S_pool_X[start:end].cpu().numpy(), sub_y.cpu().numpy())
            F_i.append(clf)

        # --- Step 3c: Setup BetweenThresholds parameters ---
        t_u = 0.5 + self.current_alpha * 0.5
        t_l = 0.5 - self.current_alpha * 0.5
        
        # Privacy budget for BetweenThresholds
        c_i = 64 * self.current_alpha * R_i 
        
        # Instantiate privacy mechanism
        bt = BetweenThresholds(
            epsilon=self.epsilon,
            delta=self.delta,
            t_l=t_l,
            t_u=t_u,
            max_T_budget=c_i,
            n=required_size,
        )

        # --- Step 3d: Query Answering Loop ---
        predictions = []
        D_i_X = [] # To store queries
        D_i_y = [] # To store predicted labels for future training
        
        processed_count = 0
        
        for x_query in query_stream:
            if processed_count >= R_i:
                break
            
            x_query = x_query.reshape(1, -1)
            
            # 3d.ii: Compute vote q(F_i)
            # The paper sums votes. To compare with thresholds ~0.5, we normalize by T_i.
            votes = sum(clf.predict(x_query)[0] for clf in F_i)
            normalized_vote = votes / T_i
            
            # 3d.ii: Get noisy outcome
            outcome = bt.query(normalized_vote)
            
            # 3d.iii: Interpret outcome
            if outcome == 'L':
                pred = 0
            elif outcome == 'R' or outcome == 'T':
                pred = 1
            else:
                # This happens if budget exhausted
                print("Privacy budget exhausted. Round halting.")
                break
            
            predictions.append(pred)
            D_i_X.append(x_query[0])
            D_i_y.append(pred)
            processed_count += 1
            
            if outcome == 'HALT':
                break

        # --- Step 3f: LabelBoost and Next Round Prep ---
        if len(D_i_X) > 0:
            # Combine S_i (labeled) and D_i (labeled by predictor)
            # The paper uses LabelBoost to "relabel" D_i to be consistent with C.
            # We approximate LabelBoost:
            # 1. Generate candidate hypotheses using S_i.
            # 2. Select best h via Exponential Mechanism based on S_i accuracy.
            # 3. Use h to label D_i for the next round.
            
            S_next_X, S_next_y = self._approximate_label_boost(
                S_X=S_pool_X, S_y=S_pool_y,
                D_X=torch.from_numpy(np.array(D_i_X)).float().to(self.device)
            )
            
            self.S_current_X = S_next_X # Already a tensor
            self.S_current_y = S_next_y
        
        # --- Step 3g: Update parameters ---
        self.current_alpha /= 2.0
        self.current_beta /= 2.0
        self.round_counter += 1
        
        return predictions

    def _approximate_label_boost(self, S_X, S_y, D_X):
        """
        Approximation of Algorithm LabelBoost.
        Instead of iterating all dichotomies, we pick the best hypothesis from 
        a candidate pool using the Exponential Mechanism.
        """
        # 1. Generate candidate set H (bootstrapping)
        candidates = []
        n_candidates = 10 if self.practical_mode else 50
        for _ in range(n_candidates):
            # Resample using torch indices
            indices = torch.randint(0, S_X.shape[0], (S_X.shape[0],), device=self.device)
            X_res, y_res = S_X[indices], S_y[indices]

            if len(torch.unique(y_res)) < 2: continue
            h = clone(self.base_learner)
            if hasattr(h, 'set_device'):
                h.set_device(self.device)

            h.fit(X_res.cpu().numpy(), y_res.cpu().numpy())
            candidates.append(h)
            
        # Exponential Mechanism
        # Score function u(h, S) = -error_S(h)
        # Probability ~ exp(epsilon * score / 2*sensitivity)
        # Sensitivity of error count is 1.
        if not candidates: # Handle case where no candidates were generated
            return D_X, torch.randint(0, 2, (D_X.shape[0],), device=self.device, dtype=torch.long)

        S_X_np = S_X.cpu().numpy()
        S_y_np = S_y.cpu().numpy()

        # Calculate scores using a list comprehension then convert to tensor
        scores = torch.tensor([
            -1 * np.sum(h.predict(S_X_np) != S_y_np) for h in candidates
        ], device=self.device, dtype=torch.float)

        # Scaled for numerical stability
        scaled_scores = (self.epsilon * scores) / 2.0
        probs = torch.softmax(scaled_scores, dim=0)
        
        # Select h
        best_h_idx = torch.multinomial(probs, 1).item()
        best_h = candidates[best_h_idx]
        
        # Relabel D and S using h
        # Step 3f says S_{i+1} comes from D'_i (the relabeled queries)
        # We return the relabeled queries as the new training set.
        new_labels_np = best_h.predict(D_X.cpu().numpy())
        return D_X, torch.from_numpy(new_labels_np).long().to(self.device)

    def auto_set_epsilon(self, X_shape, alpha=None, beta=None, safety_factor=5.0, force_minimum=False):
        """
        Automatically calculates the minimum epsilon required for the algorithm to 
        function (produce utility) given the dataset size.
        
        :param X_shape: The shape of the training data (N, features).
        :param safety_factor: Multiplier to ensure signal is stronger than noise (default 5x).
        :return: A suggested epsilon value.
        """
        N = X_shape[0]
        alpha = alpha if alpha else self.alpha
        beta = beta if beta else self.beta
        
        # 1. Calculate samples needed per teacher (lambda)
        # [cite_start]Based on Step 1 formula [cite: 235]
        alpha_1 = alpha / 2.0
        beta_1 = beta / 2.0
        term_vc = 8 * self.vc_dim * np.log(13 / alpha_1)
        term_beta = 4 * np.log(2 / beta_1)
        lambda_i = (term_vc + term_beta) / alpha_1
        
        # 2. Calculate number of teachers (T)
        # [cite_start]Based on Step 3a [cite: 241]
        if force_minimum and lambda_i > N:
            print(f"Warning: Forcing operation with small dataset ({N} < {int(lambda_i)})")
            T = 1
            lambda_i = N
            
        T = int(N / lambda_i)
        
        # 3. Calculate minimum epsilon per step
        # The noise scale in BetweenThresholds is roughly 6 / (epsilon * T).
        # We need the Noise << Signal Gap (2 * alpha).
        # Condition: (6 / (eps * T)) * safety_factor <= 2 * alpha
        
        required_epsilon = (3.0 * safety_factor) / (alpha * T)
        
        print(f"Auto-tuning Epsilon for N={N}:")
        print(f"  - Samples per Teacher (lambda): {int(lambda_i)}")
        print(f"  - Total Teachers (T): {T}")
        print(f"  - Required Signal Gap: {2*alpha:.2f}")
        print(f"  - Suggested Minimum Epsilon: {required_epsilon:.2f}")
        
        self.epsilon = required_epsilon
        return required_epsilon


class DummyClassifier:
    """Helper for degenerate cases in subsampling."""
    def __init__(self, label):
        self.label = label
    def predict(self, X):
        return np.full(len(X), self.label)
