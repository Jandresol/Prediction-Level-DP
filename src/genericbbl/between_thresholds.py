import torch

class BetweenThresholds:
    """
    Algorithm BetweenThresholds implementation using PyTorch.
    Uses numeric outcomes for efficiency.
    """
    OUTCOME_L = 0
    OUTCOME_T = 1
    OUTCOME_R = 2
    OUTCOME_HALT = 3

    def __init__(self, epsilon, delta, t_l, t_u, max_T_budget, n, device=None):
        self.epsilon = epsilon
        self.t_l = t_l
        self.t_u = t_u
        self.T_budget = max_T_budget
        self.T_count = 0
        self.n = n
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        scale = 2.0 / self.epsilon / self.n
        laplace_dist = torch.distributions.laplace.Laplace(
            torch.tensor(0.0, device=self.device), 
            torch.tensor(scale, device=self.device)
        )
        self.hat_t_l = self.t_l + laplace_dist.sample()
        self.hat_t_u = self.t_u - laplace_dist.sample()

    def query(self, q_val):
        """Processes a single query value q(S)."""
        if self.T_count >= self.T_budget:
            return self.OUTCOME_HALT

        scale = 6.0 / self.epsilon / self.n
        laplace_dist = torch.distributions.laplace.Laplace(
            torch.tensor(0.0, device=self.device), 
            torch.tensor(scale, device=self.device)
        )
        
        if not isinstance(q_val, torch.Tensor):
            q_val = torch.tensor(q_val, device=self.device)

        noisy_q = q_val + laplace_dist.sample()

        if noisy_q < self.hat_t_l:
            return self.OUTCOME_L
        elif noisy_q > self.hat_t_u:
            return self.OUTCOME_R
        else:
            self.T_count += 1
            return self.OUTCOME_T
            
    def query_batch(self, q_vals, stateless=False):
        """
        Processes a batch of query values q(S) with vectorized operations.
        
        :param q_vals: A 1D tensor or list of query values.
        :param stateless: If True, performs a fully parallel but stateless query 
                          without T_count updates or budget limits.
        :return: A tensor of outcome integers.
        """
        if not isinstance(q_vals, torch.Tensor):
            q_vals = torch.tensor(q_vals, device=self.device, dtype=torch.float32)

        batch_size = q_vals.shape[0]
        if batch_size == 0:
            return torch.tensor([], device=self.device, dtype=torch.int8)

        scale = 6.0 / self.epsilon / self.n
        laplace_dist = torch.distributions.laplace.Laplace(
            torch.zeros(batch_size, device=self.device), 
            torch.full((batch_size,), scale, device=self.device)
        )
        noisy_q_vals = q_vals + laplace_dist.sample()

        if stateless:
            outcomes = torch.full((batch_size,), self.OUTCOME_T, device=self.device, dtype=torch.int8)
            outcomes[noisy_q_vals < self.hat_t_l] = self.OUTCOME_L
            outcomes[noisy_q_vals > self.hat_t_u] = self.OUTCOME_R
            return outcomes
        else:
            is_L = noisy_q_vals < self.hat_t_l
            is_R = noisy_q_vals > self.hat_t_u
            
            outcomes = []
            for i in range(batch_size):
                if self.T_count >= self.T_budget:
                    outcomes.extend([self.OUTCOME_HALT] * (batch_size - i))
                    break

                if is_L[i]:
                    outcomes.append(self.OUTCOME_L)
                elif is_R[i]:
                    outcomes.append(self.OUTCOME_R)
                else:
                    self.T_count += 1
                    outcomes.append(self.OUTCOME_T)
            
            return torch.tensor(outcomes, device=self.device, dtype=torch.int8)
