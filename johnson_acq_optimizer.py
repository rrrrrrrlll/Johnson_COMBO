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
        assert n_suggestions == 1, "JohnsonAcqOptimizer currently only supports n_suggestions=1"
        if acq_evaluate_kwargs is None: acq_evaluate_kwargs = {}
        
        if x.dim() == 1:
            x = x.view(1, -1)

        best_x = x[0:1].clone()
        
        best_acq_val = acq_func(best_x, model=model, **acq_evaluate_kwargs)
        
        current_x = best_x

        # Hill Climbing Loop
        for i in range(self.n_steps): 
            candidates = self._get_swap_neighbors(current_x, n_neighbors=50)
            
            # FIX: Pass model explicitly
            acq_vals = acq_func(candidates, model=model, **acq_evaluate_kwargs)
            max_idx = torch.argmax(acq_vals)
            
            if acq_vals[max_idx] > best_acq_val.max():
                best_acq_val = acq_vals[max_idx]
                best_x = candidates[max_idx].unsqueeze(0)
                current_x = best_x
                
        return best_x


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