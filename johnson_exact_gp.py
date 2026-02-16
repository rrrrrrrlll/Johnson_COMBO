import gpytorch # type: ignore
from typing import Optional, Tuple
from mcbo.models.gp.exact_gp import ExactGPModel # type: ignore

from johnson_kernels import JohnsonDiffusionKernel, JohnsonRBFKernel # type: ignore


class JohnsonExactGPModel(ExactGPModel):
    """
    The unified model class.
    Inherits from MCBO's ExactGPModel wrapper.
    Internally builds the Johnson-graph-based GPyTorch model.
    """

    def __init__(self, search_space, k_size, n_size=None, diffusion: bool = False, **kwargs):
        
        if diffusion:
            kernel = gpytorch.kernels.ScaleKernel(JohnsonDiffusionKernel(k=k_size, n=n_size))
        else:
            kernel = gpytorch.kernels.ScaleKernel(JohnsonRBFKernel(k=k_size))

        super().__init__(search_space=search_space, num_out=1, kernel=kernel, **kwargs)
