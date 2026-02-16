import pandas as pd
import numpy as np
import torch # type: ignore
from typing import List, Dict, Any, Optional, Callable
from mcbo.tasks.task_base import TaskBase # type: ignore

from target_stats import get_est_stats

DEVICE = torch.device("cpu")
print(f"Running on device: {DEVICE}")

class SubgroupLabeling(TaskBase):

    def __init__(self, n, k, data: pd.DataFrame, target_stats: Dict):
        super().__init__()
        self.n = n
        self.k = k
        self.data = data
        self.stats = target_stats

    @property
    def name(self) -> str:
        return 'Subgroup labeling'

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        params = [{'name': f'g{i+1}', 'type': 'nominal', 'categories': [0, 1]} for i in range(self.n)]
        return params

    @property
    def input_constraints(self) -> Optional[List[Callable[[Dict], bool]]]:
        def check(x: Dict) -> bool:
            x_list = list(x.values())
            k_count = x_list.count(1)
            return k_count == self.k
        return [check]
    
    def evaluate(self, x: pd.DataFrame, eps=1e-8, na_penalty=1.0):
        """
        Computes the relative worst loss between estimated and target statistics.
        """
        from target_stats import get_est_stats

        # 1. Get Estimation of Statstics
        clean_targets = {k: v for k, v in self.stats.items() if v is not None}
        targets = pd.Series(clean_targets)
        est_df = get_est_stats(self.data, label=x.to_numpy(), stats=self.stats.keys())

        # 2. Identify Intersection of Columns
        # Only compare keys present in both the estimation DataFrame and the targets
        common_cols = est_df.columns.intersection(targets.index)
        
        if len(common_cols) == 0:
            raise ValueError('No shared stat between estimated and target stats')

        # 3. Align Data for Vectorized Calculation
        est_subset = est_df[common_cols]
        target_subset = targets[common_cols]

        # 4. Calculate Denominator
        denom = target_subset.abs().clip(lower=eps)

        # 5. Calculate Raw Loss
        diff = (est_subset - target_subset).abs()
        loss_matrix = diff / denom

        # 6. Apply Penalties for Non-Finite Values
        # Note: numpy isnan/isinf checks are element-wise
        est_na = ~np.isfinite(est_subset)
        target_na = ~np.isfinite(target_subset) # Series
        
        loss_matrix = loss_matrix.replace([np.inf, -np.inf], na_penalty)
        loss_matrix = loss_matrix.fillna(na_penalty)
        loss_matrix[est_na] = na_penalty
        loss_matrix[[col for col in common_cols if target_na[col]]] = na_penalty
        
        # 7. Compute Worst Loss Per Estimation
        return loss_matrix.max(axis=1)