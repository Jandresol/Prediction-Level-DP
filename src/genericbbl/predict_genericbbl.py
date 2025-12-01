import numpy as np
import torch
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

from .between_thresholds import BetweenThresholds
import concurrent.futures
import os
import multiprocessing

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
        T_size_factor = np.log(lambda_i / (self.epsilon * alpha * beta * self.delta))
        T_numerator = tau * lambda_i * np.log(1/self.delta) * T_size_factor * T_size_factor
        T_denominator = alpha * self.epsilon
        T_i = int(np.ceil(T_numerator / T_denominator))
        
        if self.practical_mode:
            # Hard clamp for running
            T_i = max(2, int(T_i / 1e7)) 
            lambda_i = max(10, int(lambda_i / 10))

        # R_i: Number of queries to answer in this round
        # S_i size is approx T_i * lambda_i
        size_S_i = T_i * lambda_i
        R_i = int((25600 * size_S_i) / self.epsilon)
        
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

    def eval(self):
        """
        Dummy method to be compatible with attack evaluation scripts, which expect a model
        with an `eval()` method. The PrivateEverlastingPredictor does not have distinct
        training/evaluation modes in the same way a single PyTorch model does.
        """
        # The base learner is cloned and fitted inside the training process, so
        # setting the top-level base_learner to eval mode has no effect.
        pass

    def __call__(self, x):
        """
        Makes the predictor callable, returning a confidence score (normalized vote)
        for a given input batch `x`. This is required for compatibility with attack scripts.
        """
        if not hasattr(self, 'F_i') or not self.F_i:
            raise RuntimeError(
                "Teacher models (F_i) are not trained. "
                "Ensure `prepare_for_simulation()` is called after `train_initial()`."
            )

        # The PyTorchCNNWrapper expects a numpy array.
        # The input `x` from the attack script is a torch tensor on the GPU.
        x_np = x.cpu().numpy()

        # Reshape if it's flat, for compatibility with CNN base learners
        if len(x_np.shape) == 2 and x_np.shape[1] == 3072:
             x_np = x_np.reshape(-1, 3, 32, 32)
        
        # This logic is adapted from `do_simulation` to get predictions from all teachers.
        tasks = [(clf, x_np) for clf in self.F_i]
        
        all_preds = []
        # Use the same parallel execution logic as do_simulation
        if self.device.type == 'cuda':
            with concurrent.futures.ThreadPoolExecutor() as executor:
                all_preds = list(executor.map(_predict_single_model_parallel, tasks))
        else:
            num_workers = min(os.cpu_count(), len(tasks))
            context = multiprocessing.get_context("spawn") if "spawn" in multiprocessing.get_all_start_methods() else None
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as executor:
                all_preds = list(executor.map(_predict_single_model_parallel, tasks))

        all_preds_np = np.array(all_preds)  # Shape: (T_i, batch_size)

        # Calculate normalized votes, which serve as the confidence score
        normalized_votes = np.sum(all_preds_np, axis=0) / self.T_i  # Shape: (batch_size,)

        # Return as a torch tensor on the correct device, with shape (batch_size, 1)
        return torch.from_numpy(normalized_votes).float().to(self.device).view(-1, 1)



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
    
    def prepare_for_simulation(self):
        """Prepares internal state for simulation mode.
        
        The teacher models will be trained in advance.
        """
        lambda_i, T_i, R_i = self._calculate_params(self.current_alpha, self.current_beta)
        
        print(f"--- Preparing for Simulation (Round {self.round_counter}) ---")
        print(f"Params: lambda={lambda_i}, T={T_i}, Max Queries R={R_i}")

        # Store for do_simulation
        self.T_i = T_i
        self.R_i = R_i
        self.lambda_i = lambda_i

        # --- Step 3a: Divide S_i into T_i disjoint databases ---
        current_size = self.S_current_X.shape[0]
        required_size = self.T_i * self.lambda_i
        
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

        # Store for later use in label boost
        self.S_pool_X = S_pool_X
        self.S_pool_y = S_pool_y

        # --- Step 3b: Train Ensemble F_i ---
        print(f"Training {self.T_i} teacher models...")

        # Parallel training is safer and more effective on CPU.
        # GPU training in parallel processes can be complex to manage (CUDA contexts, memory).
        if self.device.type == 'cpu':
            tasks = []
            for t in range(self.T_i):
                start, end = t * self.lambda_i, (t + 1) * self.lambda_i
                X_subset_np = S_pool_X[start:end].cpu().numpy()
                y_subset_np = S_pool_y[start:end].cpu().numpy()
                # Pass device as a string for pickling
                tasks.append((self.base_learner, X_subset_np, y_subset_np, 'cpu'))
            
            # Use as many workers as there are CPUs, up to the number of tasks
            num_workers = min(os.cpu_count(), len(tasks))
            print(f"Using ProcessPoolExecutor with {num_workers} workers on CPU.")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Using map to preserve order
                self.F_i = list(executor.map(_train_single_model_parallel, tasks))

        else: # Fallback to sequential training for GPU to avoid CUDA issues
            print("Device is CUDA, training models sequentially.")
            self.F_i = []
            for t in range(self.T_i):
                start, end = t * self.lambda_i, (t + 1) * self.lambda_i
                
                clf = clone(self.base_learner)
                # The wrapper will use the device it was initialized with
                
                sub_y = S_pool_y[start:end]
                if len(torch.unique(sub_y)) < 2:
                     clf = DummyClassifier(sub_y[0].item())
                else:
                    clf.fit(S_pool_X[start:end].cpu().numpy(), sub_y.cpu().numpy())
                self.F_i.append(clf)
        
        print("Teacher models trained.")

        # --- Step 3c: Setup BetweenThresholds parameters ---
        self.t_u = 0.5 + self.current_alpha * 0.5
        self.t_l = 0.5 - self.current_alpha * 0.5
        
        # Privacy budget for BetweenThresholds
        c_i = 64 * self.current_alpha * self.R_i 
        
        # Instantiate privacy mechanism
        self.bt = BetweenThresholds(
            epsilon=self.epsilon,
            delta=self.delta,
            t_l=self.t_l,
            t_u=self.t_u,
            max_T_budget=c_i,
            n=required_size,
            device=self.device,
        )

        # Initialize for do_simulation
        self.D_i_X = []
        self.D_i_y = []
        self.processed_count = 0

    def do_simulation(self, batch):
        """
        Performs one batch of simulation using pre-trained teachers.
        The prediction part is parallelized over the batch.

        :param batch: A batch of unlabeled query points x.
        :return: List of predictions (0 or 1) for the batch.
        """
        if not hasattr(self, 'F_i') or not self.F_i:
            raise RuntimeError("`prepare_for_simulation` must be called before `do_simulation`.")

        # Convert to numpy array if it's a list
        if isinstance(batch, list):
            if not batch:
                return []
            batch = np.array(batch)
        elif len(batch) == 0:
            return []

        # The expensive part is getting predictions from T_i models.
        # We can parallelize this part.
        tasks = [(clf, batch) for clf in self.F_i]
        
        all_preds = []
        if self.device.type == 'cuda':
            # For CUDA, use ThreadPoolExecutor to avoid context issues.
            # The GIL is released during GPU computation.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                all_preds = list(executor.map(_predict_single_model_parallel, tasks))
        else:  # cpu
            # For CPU, use ProcessPoolExecutor for true parallelism.
            num_workers = min(os.cpu_count(), len(tasks))
            # Using 'spawn' context is safer for libraries that use CUDA internally,
            # even when we are targeting CPU execution.
            context = multiprocessing.get_context("spawn") if "spawn" in multiprocessing.get_all_start_methods() else None
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as executor:
                all_preds = list(executor.map(_predict_single_model_parallel, tasks))

        all_preds_np = np.array(all_preds)  # Shape: (T_i, batch_size)

        # Calculate normalized votes for each query in the batch
        normalized_votes = np.sum(all_preds_np, axis=0) / self.T_i  # Shape: (batch_size,)

        # Respect the round's query limit (R_i)
        queries_left_in_round = self.R_i - self.processed_count
        if queries_left_in_round <= 0:
            print("Max queries for this round already reached.")
            return []
        
        # Only process up to the number of queries left
        votes_to_process = normalized_votes[:queries_left_in_round]
        batch_to_process = batch[:queries_left_in_round]

        # Process the allowed batch of votes at once.
        # Using stateless=True as per user request for parallel processing,
        # which assumes no T_budget exhaustion during the batch.
        outcomes = self.bt.query_batch(votes_to_process, stateless=True)
        
        # Vectorized processing of outcomes
        # pred = 0 if outcome is L (0), 1 if T (1) or R (2)
        predictions_tensor = (outcomes > BetweenThresholds.OUTCOME_L).int()
        
        # Convert to list for return value and state update
        predictions = predictions_tensor.cpu().tolist()
        
        # Update state in batch
        # self.D_i_X is a list of numpy arrays, so we convert the batch slice to a list
        self.D_i_X.extend(list(batch_to_process))
        self.D_i_y.extend(predictions)
        self.processed_count += len(predictions)
                
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
            if outcome == BetweenThresholds.OUTCOME_L:
                pred = 0
            elif outcome == BetweenThresholds.OUTCOME_R or outcome == BetweenThresholds.OUTCOME_T:
                pred = 1
            else:  # This happens if budget exhausted (OUTCOME_HALT)
                print("Privacy budget exhausted. Round halting.")
                break
            
            predictions.append(pred)
            D_i_X.append(x_query[0])
            D_i_y.append(pred)
            processed_count += 1

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

    def finalize_round(self):
        """
        Finalizes the current round after all simulations are done.
        This includes running LabelBoost and preparing for the next round.
        """
        if not hasattr(self, 'D_i_X') or not self.D_i_X:
            print("No queries were processed in this round. Nothing to finalize.")
        else:
            print("Finalizing round: relabeling queries with LabelBoost for next round's training set.")
            S_next_X, S_next_y = self._approximate_label_boost(
                S_X=self.S_pool_X, S_y=self.S_pool_y,
                D_X=torch.from_numpy(np.array(self.D_i_X)).float().to(self.device)
            )
            self.S_current_X = S_next_X
            self.S_current_y = S_next_y
        
        self.current_alpha /= 2.0
        self.current_beta /= 2.0
        self.round_counter += 1

        # Clean up state from the completed round
        if hasattr(self, 'F_i'): del self.F_i
        if hasattr(self, 'bt'): del self.bt
        if hasattr(self, 'D_i_X'): del self.D_i_X
        if hasattr(self, 'D_i_y'): del self.D_i_y
        if hasattr(self, 'processed_count'): del self.processed_count
        if hasattr(self, 'T_i'): del self.T_i
        if hasattr(self, 'R_i'): del self.R_i
        if hasattr(self, 'lambda_i'): del self.lambda_i
        if hasattr(self, 'S_pool_X'): del self.S_pool_X
        if hasattr(self, 'S_pool_y'): del self.S_pool_y
    
        print(f"--- Round {self.round_counter-1} finalized. Ready for round {self.round_counter}. ---")


class DummyClassifier:
    """Helper for degenerate cases in subsampling."""
    def __init__(self, label):
        self.label = label
    def predict(self, X):
        return np.full(len(X), self.label)


def _train_single_model_parallel(args):
    """Helper for parallel training. Can be pickled."""
    base_learner_template, X_train, y_train, device_str = args
    
    # Create a new learner instance for this process
    learner = clone(base_learner_template)
    
    # Assign the device for this worker.
    # This assumes the learner has a public `device` attribute that `fit` will use.
    # PyTorchCNNWrapper does have this.
    learner.device = device_str
            
    # Handle single-class edge case in subsample
    # Need to convert from numpy back to tensor to use torch.unique
    if len(np.unique(y_train)) < 2:
         return DummyClassifier(y_train[0]) # .item() is not needed for numpy scalar
    else:
        learner.fit(X_train, y_train)
    return learner


def _predict_single_model_parallel(args):
    """Helper for parallel prediction."""
    clf, batch = args
    return clf.predict(batch)


def train_genericbbl(
    train_data,
    test_data,
    batch_size=128,
    epochs=10,
    lr=1e-3,
    epsilon=75,
    target_delta=1e-5,
    save_dir="./results/metrics",
    eval=True
):
    """
    Initializes and evaluates the GenericBBL private everlasting predictor.

    This function sets up the PrivateEverlastingPredictor with a given base learner,
    trains it on an initial dataset, and then evaluates its performance on a stream
    of test data. The evaluation proceeds in rounds, as defined by the GenericBBL
    algorithm, until the entire test set is processed or the privacy budget is
    exhausted.

    The performance metrics, including accuracy and runtime, are saved to a JSON file.

    Args:
        train_data (dict): A dictionary containing the initial training data 
                           (e.g., {'images': tensor, 'labels': tensor}).
        test_data (dict): A dictionary containing the test data for evaluation.
        batch_size (int, optional): The batch size for the base learner's training
                                    and for processing the evaluation stream. Defaults to 128.
        epochs (int, optional): The number of training epochs for each base learner (teacher model).
                                Defaults to 10.
        lr (float, optional): The learning rate for the base learner's optimizer.
                              Defaults to 1e-3.
        epsilon (float, optional): The total privacy budget ε for the GenericBBL predictor.
                                   Defaults to 75.
        noise_multiplier (float, optional): Included for API consistency with other training
                                            functions. Not directly used by GenericBBL.
        max_grad_norm (float, optional): Included for API consistency. Not directly used.
        target_delta (float, optional): The privacy parameter δ for the GenericBBL predictor.
                                        Defaults to 1e-5.
        save_dir (str, optional): The directory to save the final metrics JSON file.
                                  Defaults to "./results/metrics".
        eval (bool, optional): If False, skips the evaluation phase. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - PrivateEverlastingPredictor: The trained predictor instance.
            - float: The final accuracy on the test set (or None if eval is False).
            - float: The total epsilon privacy budget used.
    """
    from src.genericbbl.pytorch_wrapper import PyTorchCNNWrapper
    from src.models.cifar_cnn import cifar_cnn
    import time
    import os
    import json
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Setup Predictor
    start_time = time.time()
    
    X_initial = train_data["images"].float() / 255.0
    y_initial = train_data["labels"].float()

    predictor = PrivateEverlastingPredictor(
        base_learner=PyTorchCNNWrapper(
            model_class=cifar_cnn, epochs=epochs, batch_size=batch_size, lr=lr, device=str(device)),
        vc_dim=20,
        epsilon=epsilon,
        delta=target_delta,
        alpha=0.2,
        beta=0.2,
        practical_mode=True,
        device=device,
    )
    # The wrapper expects numpy arrays, so we move to CPU first.
    predictor.train_initial(X_initial.cpu().numpy(), y_initial.cpu().numpy())
    
    training_time = time.time() - start_time
    
    if not eval:
        print(f"Training completed in {training_time:.2f} seconds.")
        # When eval is False, it is likely being called from an attack script.
        # We need to prepare the teacher models so the predictor can be called like a regular model.
        print("Preparing teacher models for attack evaluation...")
        predictor.prepare_for_simulation()
        return predictor, None, epsilon

    # 2. Evaluation
    print("\n--- Starting Evaluation ---")
    eval_start_time = time.time()
    
    X_test = test_data["images"].float() / 255.0
    y_test = test_data["labels"]
    
    # The wrapper expects numpy arrays.
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    all_preds = []
    total_samples = len(X_test_np)
    processed_samples = 0
    predictor.prepare_for_simulation()

    while processed_samples < total_samples:
        print(f"\nStarting predictor round {predictor.round_counter}")

        queries_this_round = predictor.R_i
        start_idx = processed_samples
        end_idx = min(processed_samples + queries_this_round, total_samples)
        
        if start_idx >= end_idx:
            print("No more samples to process for evaluation.")
            break
            
        query_stream = X_test_np[start_idx:end_idx]
        
        round_preds = []
        for i in range(0, len(query_stream), batch_size):
            batch_X = query_stream[i:i+batch_size]
            batch_preds = predictor.do_simulation(batch_X)
            round_preds.extend(batch_preds)

        all_preds.extend(round_preds)
        processed_samples += len(round_preds)

        print(f"Round {predictor.round_counter} completed. Processed {len(round_preds)} queries.")
        
        # predictor.finalize_round()

        if not round_preds or len(round_preds) < len(query_stream):
            print("Stopping evaluation as round ended early (budget exhausted or query limit).")
            break

    eval_runtime = time.time() - eval_start_time
    total_runtime = time.time() - start_time
    
    y_true = y_test_np[:len(all_preds)]
    accuracy = np.mean(np.array(all_preds) == y_true) if len(all_preds) > 0 else 0.0
    print(f"\nEvaluation Complete. Accuracy: {accuracy:.4f}")

    # 3. Save Metrics
    metrics = {
        "dataset": "CIFAR-10 binary",
        "epochs_per_teacher": epochs,
        "lr_teacher": lr,
        "batch_size_eval": batch_size,
        "epsilon_bbl": epsilon,
        "delta_bbl": target_delta,
        "accuracy": accuracy,
        "total_runtime_sec": total_runtime,
        "training_time_sec": training_time,
        "evaluation_time_sec": eval_runtime,
        "num_rounds_eval": predictor.round_counter - 1,
        "num_samples_evaluated": len(all_preds),
    }

    os.makedirs(save_dir, exist_ok=True)
    filename = "genericbbl_cifar10.json"
    filepath = os.path.join(save_dir, filename)
    json.dump(metrics, open(filepath, "w"), indent=4)
    print(f"Saved metrics to {filepath}")

    return predictor, accuracy, epsilon
