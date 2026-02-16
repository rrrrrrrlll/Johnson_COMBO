import numpy as np
import pandas as pd
from typing import Optional, Tuple
import torch # type: ignore
import gpytorch # type: ignore
from gpytorch.kernels import Kernel # type: ignore
from gpytorch.constraints import Positive # type: ignore
from scipy.special import comb


class JohnsonDiffusionKernel(Kernel):
    def __init__(self, n, k, **kwargs):
        super(JohnsonDiffusionKernel, self).__init__(**kwargs)
        self.n = n
        self.k = k
        
        # 1. Manually Register 'raw_beta'
        # We use a scalar (1,) so it broadcasts safely against (N, N)
        self.register_parameter(
            name="raw_beta", 
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        
        # 2. Manually Register Constraint (Softplus)
        self.register_constraint("raw_beta", Positive())

        # Cache for lookup table (calculated on CPU)
        self._lookup_table_cache = None
        self._last_beta_value = None

    @property
    def beta(self):
        # Helper to get the actual positive beta value
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        # Helper to set beta (useful for initialization)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def _compute_lookup_table(self):
        # 1. Get current beta value (scalar float)
        beta_val = self.beta.item()
        
        # Return cached table if beta hasn't changed
        if self._last_beta_value == beta_val and self._lookup_table_cache is not None:
            return self._lookup_table_cache

        # --- Compute Spectral Kernel on CPU ---
        j = np.arange(self.k + 1)
        
        # Eigenvalues: lambda_j = k(n-k) - mu_j
        mu_j = (self.k - j) * (self.n - self.k - j) - j
        lambda_j = (self.k * (self.n - self.k)) - mu_j
        
        # Multiplicities
        m_j = comb(self.n, j) - comb(self.n, j - 1)
        
        # Decay Factors
        decay = np.exp(-beta_val * lambda_j)
        
        # Eberlein Polynomials (Matrix E[j, i])
        E = np.zeros((self.k + 1, self.k + 1))
        for j_idx in range(self.k + 1):
            for i_idx in range(self.k + 1):
                total = 0.0
                for r in range(j_idx + 1):
                    if (self.k - i_idx - r >= 0) and (i_idx - r >= 0):
                        term = ((-1)**r) * comb(j_idx, r) * \
                               comb(self.k - j_idx, self.k - i_idx - r) * \
                               comb(self.n - self.k - j_idx, i_idx - r)
                        total += term
                E[j_idx, i_idx] = total

        # Weighted Sum
        total_vertices = comb(self.n, self.k)
        weighted_modes = decay * m_j
        K_lookup_np = np.dot(weighted_modes, E) / total_vertices
        
        # Move to GPU/Device matching the parameter
        K_lookup = torch.tensor(
            K_lookup_np, 
            dtype=self.raw_beta.dtype, 
            device=self.raw_beta.device
        )
        
        self._lookup_table_cache = K_lookup
        self._last_beta_value = beta_val
        
        return K_lookup

    def forward(self, x1, x2, diag=False, **params):
        # 1. Handle Symmetric Case (x2 is None)
        if x2 is None:
            x2 = x1
            
        # 2. Calculate Intersections
        # Result shape: (Batch, N, M) or (N, M)
        if diag:
            # If diag=True, we only want the diagonal elements.
            # In GPyTorch 1.6, we must return a tensor of shape matching input batch.
            # For Johnson graph, K(x,x) is always the max value (intersection = k).
            
            # Compute the table to get the max value
            lookup_table = self._compute_lookup_table()
            max_val = lookup_table[-1] # Index k is self-intersection
            
            return max_val.expand(x1.shape[:-1]) # Expand to (Batch, N)

        else:
            # Full Matrix
            intersections = torch.matmul(x1, x2.transpose(-1, -2))
            
            # 3. Lookup Values
            # Clamp allows us to safely handle floating point errors (e.g. 2.999 -> 2)
            indices = intersections.round().long().clamp(0, self.k)
            
            # 4. Return from Table
            lookup_table = self._compute_lookup_table()
            return lookup_table[indices]


class JohnsonRBFKernel(Kernel):
    def __init__(self, k, **kwargs):
        super(JohnsonRBFKernel, self).__init__(**kwargs)
        self.k = k
        
        # 1. Manually register the raw parameter
        # Shape (1, 1) ensures it works with both (N, N) and (B, N, N) matrices
        self.register_parameter(
            name="raw_lengthscale", 
            parameter=torch.nn.Parameter(torch.zeros(1, 1))
        )
        
        # 2. Manually register the constraint
        # In 1.6.0, this ensures the .raw_lengthscale_constraint attribute exists
        self.register_constraint("raw_lengthscale", Positive())

    # We skip the @property decorator to avoid 'AttributeError' confusion.
    # We will compute it locally in forward() instead.

    def forward(self, x1, x2, diag=False, **params):
        # 3. Handle Symmetric Case (Critical for GPyTorch internals)
        if x2 is None:
            x2 = x1

        # 4. Diagonal Mode (Variance)
        if diag:
            return torch.ones(
                x1.shape[:-1], 
                dtype=x1.dtype, 
                device=x1.device
            )

        # 5. Full Covariance Mode
        # Intersection: (Batch, N, M) or (N, M)
        intersections = torch.matmul(x1, x2.transpose(-1, -2))
        
        # Distance (Swaps): (Batch, N, M)
        dist = self.k - intersections
        
        # 6. Transform Lengthscale LOCALLY
        # We grab the raw param and apply the constraint manually here.
        # This bypasses any "missing property" errors.
        lengthscale = self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
        
        # 7. Apply RBF
        # lengthscale is (1, 1). dist is (N, M).
        # PyTorch will broadcast (1, 1) to (N, M) correctly.
        # We do NOT use .view(), which prevents the [1, 30, 30] error.
        
        dist_sq = dist.pow(2)
        scaled_dist = dist_sq / lengthscale.pow(2)
        
        return torch.exp(-0.5 * scaled_dist)