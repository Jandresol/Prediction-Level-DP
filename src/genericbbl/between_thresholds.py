import numpy as np

class BetweenThresholds:
    """
    Algorithm BetweenThresholds implementation.
    """
    def __init__(self, epsilon, delta, t_l, t_u, max_T_budget, practical_mode=True):
        self.epsilon = epsilon
        self.t_l = t_l
        self.t_u = t_u
        self.T_budget = max_T_budget
        self.T_count = 0
        self.practical_mode = practical_mode
        
        # Initialize noisy thresholds
        # Scale noise for implementation stability
        scale = (2.0 / epsilon) if practical_mode else (2.0 / (epsilon * 1000))
        self.hat_t_l = t_l + np.random.laplace(0, scale)
        self.hat_t_u = t_u - np.random.laplace(0, scale) # Note: paper says t_u - mu

    def query(self, q_val):
        """
        Processes a single query value q(S).
        """
        if self.T_count >= self.T_budget:
            return 'HALT'

        # Add noise to query value
        # Paper: Lap(6 / epsilon * n). Here n is implicitly handled by normalized q_val or scale
        scale = (6.0 / self.epsilon) if self.practical_mode else (6.0 / (self.epsilon * 1000))
        noisy_q = q_val + np.random.laplace(0, scale)

        # Output logic
        if noisy_q < self.hat_t_l:
            return 'L'
        elif noisy_q > self.hat_t_u:
            return 'R'
        else:
            # Output T (Top/True)
            self.T_count += 1
            return 'T'
