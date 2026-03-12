"""
Optimizer types exposed by the NeoMatrix Rust backend.

Available: GradientDescent (SGD with configurable learning rate).
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neomatrix._backend import Tensor, ParametersRef

from neomatrix._backend import GradientDescent

__all__ = [
    "GradientDescent",
    # Protocols
    "Optimizer",
]


@runtime_checkable
class Optimizer(Protocol):
    """Structural protocol for any optimizer.

    All optimizers expose ``update``, which modifies layer weights and biases
    in-place based on their gradients.

    Args:
        weights: Weight tensor to update (modified in-place).
        biases: Bias tensor to update (modified in-place).
        w_grads: Weight gradient tensor.
        b_grads: Bias gradient tensor.
        step: Current training step (used by adaptive optimizers like Adam).
    """

    def register_params(self, params: list[ParametersRef]): ...
    def step(self): ...
    def zero_grad(self): ...
