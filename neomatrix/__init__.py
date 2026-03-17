"""
NeoMatrix — ML library with Rust backend.

All symbols are available directly:
    from neomatrix import Tensor, Dense, ReLU, MSE, GradientDescent

Or via submodules for better organisation:
    from neomatrix.layers import Dense, ReLU, Sigmoid
    from neomatrix.losses import MSE, CCE
    from neomatrix.optimizers import GradientDescent, Optimizer
"""

from neomatrix import layers, losses, metrics, model, optimizers, utils
from neomatrix._backend import Tensor
from neomatrix.layers import Dense, Init, ReLU, Sigmoid, Softmax, Tanh
from neomatrix.losses import BCE, CCE, MAE, MSE, HingeLoss, HuberLoss
from neomatrix.model import Model
from neomatrix.optimizers import GradientDescent, MomentumGD, Optimizer, ParametersRef

__all__ = [
    # Core
    "Tensor",
    "Model",
    # Submodules
    "model",
    "layers",
    "losses",
    "optimizers",
    "utils",
    "metrics",
    # Layers (flat re-export)
    "Dense",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Init",
    # Losses (flat re-export)
    "MSE",
    "MAE",
    "BCE",
    "CCE",
    "HuberLoss",
    "HingeLoss",
    # Optimizers (flat re-export)
    "GradientDescent",
    "MomentumGD",
    "Optimizer",
    "ParametersRef",
]
