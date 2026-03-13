"""
NeoMatrix — ML library with Rust backend.

All symbols are available directly:
    from neomatrix import Tensor, Dense, ReLU, MSE, GradientDescent

Or via submodules for better organisation:
    from neomatrix.layers import Dense, ReLU, Sigmoid
    from neomatrix.losses import MSE, CCE
    from neomatrix.optimizers import GradientDescent, Optimizer
"""

from neomatrix._backend import Tensor

from neomatrix.model import Model
from neomatrix.layers import Dense, ReLU, Sigmoid, Tanh, Softmax, Init
from neomatrix.losses import MSE, MAE, BCE, CCE, HuberLoss, HingeLoss
from neomatrix.optimizers import GradientDescent, Optimizer, ParametersRef

from neomatrix import model, layers, losses, optimizers, utils

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
    "Optimizer",
    "ParametersRef",
]
