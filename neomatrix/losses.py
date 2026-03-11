"""
Loss functions exposed by the NeoMatrix Rust backend.

Available: MSE, MAE, BCE, CCE, HuberLoss, HingeLoss.
"""
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neomatrix._backend import Tensor

from neomatrix._backend import (
    MSE,
    MAE,
    BCE,
    CCE,
    HuberLoss,
    HingeLoss,
)

__all__ = [
    "MSE",
    "MAE",
    "BCE",
    "CCE",
    "HuberLoss",
    "HingeLoss",
    # Protocols
    "LossFunction",
    "FusedLossFunction",
]

@runtime_checkable
class LossFunction(Protocol):
    """Structural protocol for any loss function.

    All loss functions expose ``call`` (forward pass, returns scalar)
    and ``backward`` (gradient w.r.t. predictions).
    """
    def call(self, y_true: Tensor, y_pred: Tensor) -> float: ...
    def backward(self, y_true: Tensor, y_pred: Tensor) -> Tensor: ...


@runtime_checkable
class FusedLossFunction(LossFunction, Protocol):
    """Structural protocol for loss functions that support fused gradients.

    ``BCE`` and ``CCE`` expose ``backward_optimized``, which returns the
    simplified gradient ``y_pred - y_true`` (valid when the preceding
    activation is Sigmoid/Softmax respectively).
    """
    def backward_optimized(self, y_true: Tensor, y_pred: Tensor) -> Tensor: ...