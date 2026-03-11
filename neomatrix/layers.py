"""
Layer types exposed by the NeoMatrix Rust backend.

Dense layer and activation layers, plus weight initialization strategies.
"""
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neomatrix._backend import Tensor

from neomatrix._backend import (
    Dense,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Init,
)

__all__ = [
    "Dense",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Init",
    # Protocols
    "Layer",
    "TrainableLayer",
    "FusedLayer",
]

@runtime_checkable
class Layer(Protocol):
    """Structural protocol for any layer (activation or trainable).

    All layers in NeoMatrix expose ``forward`` and ``backward``.
    """
    def forward(self, input: Tensor, training: bool) -> Tensor: ...
    def backward(self, output_gradient: Tensor) -> Tensor: ...


@runtime_checkable
class TrainableLayer(Layer, Protocol):
    """Structural protocol for layers that hold learnable parameters (e.g. Dense).

    Extends :class:`Layer` with ``get_params_and_grads``, which returns
    ``[(weight, weight_grad), (bias, bias_grad)]`` after a backward pass.
    """
    def get_params_and_grads(self) -> list[tuple[Tensor, Tensor]]: ...


@runtime_checkable
class FusedLayer(Layer, Protocol):
    """Structural protocol for activation layers that support fused loss gradients.

    ``Sigmoid`` and ``Softmax`` expose ``backward_with_logits``, which skips
    the full Jacobian and directly returns the optimised gradient
    ``activation(z) - y_true`` when combined with BCE or CCE loss.
    """
    def backward_optimized(self, output_gradient: Tensor) -> Tensor: ...