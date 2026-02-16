import numpy as np
import torch # type: ignore
from typing import Optional, Dict, Any

from mcbo.models.model_base import ModelBase # type: ignore
from mcbo.acq_optimizers.acq_optimizer_base import AcqOptimizerBase  # type: ignore


class JohnsonLSAcqOptimizer(AcqOptimizerBase):

    def __init__(self, search_space, n_steps=1000, n_swaps=1, **kwargs):
        self.search_space = search_space
        self.n_swaps = n_swaps
        self.n_steps = n_steps
        super().__init__(search_space=search_space, dtype=torch.float64, **kwargs)

    @property
    def name(self): return "Johnson Local Swap Search"

    def get_color_1(self):
        return "Blue"

    def _get_swap_neighbors(self, x: torch.Tensor, n_neighbors: int = 20) -> torch.Tensor:
        x_np = x.detach().cpu().numpy().astype(int)
        current_x = x_np[0] 
        neighbors = []
        ones_idx = np.where(current_x == 1)[0]
        zeros_idx = np.where(current_x == 0)[0]
        
        if len(ones_idx) < self.n_swaps or len(zeros_idx) < self.n_swaps:
            return x.repeat(n_neighbors, 1)

        for _ in range(n_neighbors):
            x_new = current_x.copy()
            i = np.random.choice(ones_idx, size=self.n_swaps, replace=False)
            j = np.random.choice(zeros_idx, size=self.n_swaps, replace=False)
            x_new[i] = 0
            x_new[j] = 1
            neighbors.append(x_new)
            
        return torch.tensor(np.array(neighbors), dtype=x.dtype, device=x.device)

    def optimize(self,
                 x: torch.Tensor,
                 n_suggestions: int,
                 x_observed: torch.Tensor,
                 model: ModelBase,
                 acq_func: Any,
                 acq_evaluate_kwargs: Dict = None,
                 tr_manager: Any = None,
                 **kwargs
                 ) -> torch.Tensor:
        """
        Optimizes the acquisition function using Hill Climbing with Swap mutations.
        Matches AcqOptimizerBase signature.
        """

        if acq_evaluate_kwargs is None: acq_evaluate_kwargs = {}
        
        # Ensure input is 2D
        if x.dim() == 1: x = x.view(1, -1)
        N_dim = x.shape[1]

        # 1. Initialize Multiple Starting Points (Walkers)
        walkers = torch.zeros((n_suggestions, N_dim), dtype=x.dtype, device=x.device)
        
        # Walker 0: Starts at the provided best point (exploitation)
        walkers[0] = x[0]
        
        # Walkers 1..n: Start at random points from history (exploration)
        if n_suggestions > 1:
            if x_observed is not None and len(x_observed) > 0:
                # Sample random indices from history
                n_needed = n_suggestions - 1
                indices = np.random.choice(len(x_observed), size=n_needed, replace=True)
                history_samples = x_observed[indices].to(x.device)
                walkers[1:] = history_samples
            else:
                # Fallback if no history: just clone the best point
                # They will diverge later due to stochastic neighbors
                walkers[1:] = x[0].clone()

        # Track best point found by each walker
        best_walkers = walkers.clone()
        best_scores = acq_func(walkers, model=model, **acq_evaluate_kwargs).view(-1) # (n_suggestions,)
        
        # 2. Parallel Optimization Loop
        for _ in range(self.n):
            # Generate neighbors for EACH walker
            # We create a large batch of candidates: (n_suggestions * 20, N)
            candidate_batches = []
            
            for i in range(n_suggestions):
                # Generate 20 neighbors for walker i
                cands = self._get_swap_neighbors(walkers[i:i+1], n_neighbors=20)
                candidate_batches.append(cands)
            
            # Combine into one tensor for efficient GP evaluation
            all_candidates = torch.cat(candidate_batches, dim=0)
            
            # Evaluate all candidates at once
            all_scores = acq_func(all_candidates, model=model, **acq_evaluate_kwargs).view(n_suggestions, 20)
            
            # 3. Update Step
            # Find the best neighbor for each walker
            max_scores, max_idxs = torch.max(all_scores, dim=1) # (n_suggestions,)
            
            # Check for improvement
            improved = max_scores > best_scores
            
            # Update best scores where improved
            best_scores = torch.where(improved, max_scores, best_scores)
            
            # Update walker positions where improved
            for i in range(n_suggestions):
                if improved[i]:
                    new_pos = candidate_batches[i][max_idxs[i]]
                    walkers[i] = new_pos
                    best_walkers[i] = new_pos
                
        return best_walkers



# --- OPTION B: Simulated Annealing (SA) ---
class JohnsonSAAcqOptimizer(AcqOptimizerBase):

    def __init__(self, search_space, n_swaps=1, n_steps=2000, T_max=1.0, cooling=0.995, **kwargs):
        self.search_space = search_space
        self.n_swaps = n_swaps
        self.n_steps = n_steps
        self.T_max = T_max
        self.cooling = cooling
        super().__init__(search_space=search_space, dtype=torch.float64, **kwargs)

    @property
    def name(self): return "Johnson SA Search"

    def get_color_1(self):
        return "Blue"

    def _get_swap_neighbor(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy().astype(int).flatten()
        ones_idx = np.where(x_np == 1)[0]
        zeros_idx = np.where(x_np == 0)[0]
        
        if len(ones_idx) < self.n_swaps or len(zeros_idx) < self.n_swaps:
            return x

        x_new = x_np.copy()
        i = np.random.choice(ones_idx, size=self.n_swaps, replace=False)
        j = np.random.choice(zeros_idx, size=self.n_swaps, replace=False)
        x_new[i] = 0
        x_new[j] = 1
        
        return torch.tensor(x_new.reshape(1, -1), dtype=x.dtype, device=x.device)

    def optimize(self,
                 x: torch.Tensor,
                 n_suggestions: int,
                 x_observed: torch.Tensor,
                 model: ModelBase,
                 acq_func: Any,
                 acq_evaluate_kwargs: Dict = None,
                 tr_manager: Any = None,
                 **kwargs
                 ) -> torch.Tensor:
        """
        Optimizes the acquisition function using Simulated Annealing with Swap mutations.
        Matches AcqOptimizerBase signature.
        """
        assert n_suggestions == 1, "SA acquisition optimizer does not support suggesting batches of data"
        if acq_evaluate_kwargs is None: acq_evaluate_kwargs = {}

        if x.dim() == 1:
            x = x.view(1, -1)

        # 1. Initialize
        # x is the starting point (e.g., best observed or TR center)
        current_x = x[0:1].clone()
        current_val = acq_func(current_x, model=model, **acq_evaluate_kwargs).item()

        best_x = current_x.clone()
        best_val = current_val
        
        T = self.T_max
        
        # 2. Annealing Loop
        for _ in range(self.n_steps):
            # A. Generate Candidate
            candidate = self._get_swap_neighbor(current_x)
            cand_val = acq_func(candidate, model=model, **acq_evaluate_kwargs).item()
            
            # B. Metropolis Criterion (Maximization)
            diff = cand_val - current_val
            
            if diff > 0:
                accept = True
                if cand_val > best_val:
                    best_val = cand_val
                    best_x = candidate.clone()
            else:
                p = np.exp(diff / T)
                accept = np.random.rand() < p
                
            if accept:
                current_x = candidate
                current_val = cand_val
                
            # C. Cool down
            T *= self.cooling
            
        return best_x