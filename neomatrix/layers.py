"""
Layer types exposed by the NeoMatrix Rust backend.

This module re-exports layer implementations from the Rust `neomatrix-core` library
via PyO3 bindings, providing a clean Python interface for building neural networks.

**Available Layers:**

- **Dense**: Fully-connected (linear) layer with learnable weights and biases
- **ReLU**: Rectified Linear Unit activation (f(x) = max(0, x))
- **Sigmoid**: Sigmoid activation (f(x) = 1 / (1 + e^(-x)))
- **Tanh**: Hyperbolic tangent activation (f(x) = tanh(x))
- **Softmax**: Softmax activation for multi-class classification

**Weight Initialization:**

- **Init.Xavier**: Xavier/Glorot initialization (variance scaling for sigmoid/tanh)
- **Init.He**: He initialization (variance scaling for ReLU)
- **Init.LeCun**: LeCun initialization (variance scaling for SELU)
- **Init.Random**: Uniform random initialization in [-1, 1]

**Structural Protocols:**

- **Layer**: Base protocol for all layers (forward + backward methods)
- **TrainableLayer**: Protocol for layers with learnable parameters (adds get_parameters)
- **FusedLayer**: Protocol for activations supporting fused loss gradients (backward_optimized)

**Example:**

```python
from neomatrix import layers

# Build a simple network
dense1 = layers.Dense(128, 784, layers.Init.He)  # Input layer
relu = layers.ReLU()
dense2 = layers.Dense(10, 128, layers.Init.Xavier)  # Output layer
softmax = layers.Softmax()

# Check capabilities
if isinstance(dense1, layers.TrainableLayer):
    params = dense1.get_parameters()  # Returns ParametersRef for optimizer

if isinstance(softmax, layers.FusedLayer):
    # Can use fused gradient with CCE loss
    pass
```
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neomatrix._backend import Tensor, ParametersRef

from neomatrix._backend import (
    Dense,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Init,
)

# Add comprehensive docstrings to re-exported classes
Dense.__doc__ = """
Fully-connected (linear) layer with learnable weights and biases.

Computes: **Y = X·W + b**

Where:
- X: Input tensor [batch_size, in_features]
- W: Weight matrix [in_features, out_features]
- b: Bias vector [out_features]
- Y: Output tensor [batch_size, out_features]

**Constructor:**

```python
Dense(out_nodes: int, in_nodes: int, init: Init)
```

**Parameters:**
- `out_nodes`: Number of output neurons
- `in_nodes`: Number of input neurons (from previous layer)
- `init`: Weight initialization strategy (Init.He, Init.Xavier, Init.LeCun, Init.Random)

**Methods:**
- `forward(input: Tensor, training: bool) -> Tensor`: Computes Y = X·W + b
- `backward(output_gradient: Tensor) -> Tensor`: Backpropagates gradients, accumulates ∂L/∂W and ∂L/∂b
- `get_parameters() -> ParametersRef`: Returns shared references to weights/biases/gradients for optimizer

**Example:**

```python
from neomatrix import layers

# Create 128-neuron layer accepting 784 inputs (e.g., 28×28 MNIST images)
layer = layers.Dense(128, 784, layers.Init.He)

# Forward pass
output = layer.forward(input_tensor, training=True)

# Backward pass (during training)
input_grad = layer.backward(output_grad)

# Get parameters for optimizer
params = layer.get_parameters()
```

**Notes:**
- Weights initialized according to `init` strategy
- Biases initialized to zeros
- Gradients accumulated during `backward()` — optimizer must call `zero_grad()` before each step
"""

ReLU.__doc__ = """
Rectified Linear Unit (ReLU) activation layer.

**Formula:** f(x) = max(0, x)

**Gradient:** f'(x) = 1 if x > 0 else 0

ReLU is the most common activation function for hidden layers in deep networks.
It's computationally efficient and helps mitigate vanishing gradients.

**Constructor:**

```python
ReLU()
```

**Methods:**
- `forward(input: Tensor, training: bool) -> Tensor`: Applies ReLU element-wise
- `backward(output_gradient: Tensor) -> Tensor`: Backpropagates gradients (zeros out where input ≤ 0)

**Example:**

```python
from neomatrix import layers

relu = layers.ReLU()
activated = relu.forward(input_tensor, training=True)
input_grad = relu.backward(output_grad)
```

**Properties:**
- Non-saturating (no vanishing gradient for positive inputs)
- Sparse activation (typically 50% of neurons are zero)
- Computationally cheap (simple max operation)
- Can cause "dying ReLU" problem (neurons stuck at zero)
"""

Sigmoid.__doc__ = """
Sigmoid activation layer.

**Formula:** σ(x) = 1 / (1 + e^(-x))

**Gradient:** σ'(x) = σ(x) · (1 - σ(x))

**Fused Gradient (with BCE loss):** ∂L/∂z = σ(z) - y_true

Sigmoid squashes inputs to range [0, 1], making it suitable for binary classification.
However, it suffers from vanishing gradients for large |x|.

**Constructor:**

```python
Sigmoid()
```

**Methods:**
- `forward(input: Tensor, training: bool) -> Tensor`: Applies sigmoid element-wise
- `backward(output_gradient: Tensor) -> Tensor`: Standard gradient (σ(1-σ))
- `backward_optimized(output_gradient: Tensor) -> Tensor`: **Fused gradient for BCE loss** (passthrough)

**Example:**

```python
from neomatrix import layers, losses

sigmoid = layers.Sigmoid()
output = sigmoid.forward(logits, training=True)

# Standard backprop
grad = sigmoid.backward(loss_grad)

# Optimized backprop (when using BCE loss with raw logits)
if isinstance(sigmoid, layers.FusedLayer):
    grad = sigmoid.backward_optimized(bce_loss.backward_optimized(y_true, logits))
```

**Use Cases:**
- Binary classification (output layer)
- Gate mechanisms in LSTMs/GRUs
- When you need outputs in [0, 1] range

**Limitations:**
- Vanishing gradients for |x| > 5
- Not zero-centered (can slow convergence)
"""

Tanh.__doc__ = """
Hyperbolic Tangent (tanh) activation layer.

**Formula:** tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

**Gradient:** tanh'(x) = 1 - tanh²(x)

Tanh squashes inputs to range [-1, 1], making it zero-centered (unlike Sigmoid).
This can lead to faster convergence in hidden layers.

**Constructor:**

```python
Tanh()
```

**Methods:**
- `forward(input: Tensor, training: bool) -> Tensor`: Applies tanh element-wise
- `backward(output_gradient: Tensor) -> Tensor`: Backpropagates gradients

**Example:**

```python
from neomatrix import layers

tanh = layers.Tanh()
activated = tanh.forward(input_tensor, training=True)
input_grad = tanh.backward(output_grad)
```

**Properties:**
- Zero-centered (mean activation ≈ 0)
- Stronger gradients than Sigmoid (range [-1, 1] vs [0, 1])
- Still suffers from vanishing gradients for large |x|

**Use Cases:**
- Hidden layers (alternative to ReLU)
- RNNs (traditional activation choice)
- When zero-centered activations are beneficial
"""

Softmax.__doc__ = """
Softmax activation layer for multi-class classification.

**Formula:** softmax(x)ᵢ = e^(xᵢ) / Σⱼ e^(xⱼ)

**Properties:**
- Converts logits to probability distribution (all outputs sum to 1.0)
- Output range: [0, 1] per class
- Used exclusively as final activation for multi-class classification

**Gradient (standard):**
Full Jacobian matrix (computationally expensive)

**Fused Gradient (with CCE loss):**
∂L/∂z = softmax(z) - y_true (simple subtraction!)

**Constructor:**

```python
Softmax()
```

**Methods:**
- `forward(input: Tensor, training: bool) -> Tensor`: Applies softmax to convert logits to probabilities
- `backward(output_gradient: Tensor) -> Tensor`: Standard gradient (full Jacobian)
- `backward_optimized(output_gradient: Tensor) -> Tensor`: **Fused gradient for CCE loss** (passthrough)

**Example:**

```python
from neomatrix import layers, losses

softmax = layers.Softmax()
probs = softmax.forward(logits, training=True)  # Probabilities sum to 1

# Standard backprop (if loss gradient already computed)
grad = softmax.backward(loss_grad)

# Optimized backprop (when using CCE loss with raw logits)
if isinstance(softmax, layers.FusedLayer):
    grad = softmax.backward_optimized(cce_loss.backward_optimized(y_true, logits))
```

**Use Cases:**
- Multi-class classification (output layer)
- Always paired with Categorical Cross-Entropy loss
- When you need class probabilities (not just scores)

**Numerical Stability:**
Uses log-sum-exp trick internally: softmax(x) = softmax(x - max(x))
This prevents overflow from large exponentials.
"""

Init.__doc__ = """
Weight initialization strategies for Dense layers.

Proper weight initialization is critical for training deep networks. Different
activations benefit from different initialization schemes.

**Available Strategies:**

- **Init.Xavier** (Glorot): For Sigmoid/Tanh activations
  - Variance: 2 / (fan_in + fan_out)
  - Distribution: Uniform[-√6/√(fan_in+fan_out), +√6/√(fan_in+fan_out)]

- **Init.He**: For ReLU/Leaky ReLU activations
  - Variance: 2 / fan_in
  - Distribution: Uniform[-√6/√fan_in, +√6/√fan_in]

- **Init.LeCun**: For SELU activations
  - Variance: 1 / fan_in
  - Distribution: Uniform[-√3/√fan_in, +√3/√fan_in]

- **Init.Random**: Simple uniform random initialization
  - Distribution: Uniform[-1, 1]

**Example:**

```python
from neomatrix import layers

# ReLU network → use He initialization
layer1 = layers.Dense(128, 784, layers.Init.He)
relu = layers.ReLU()

# Sigmoid network → use Xavier initialization
layer2 = layers.Dense(10, 128, layers.Init.Xavier)
sigmoid = layers.Sigmoid()
```

**Guidelines:**
- **ReLU-based networks**: Use Init.He (accounts for dying ReLU problem)
- **Sigmoid/Tanh networks**: Use Init.Xavier (symmetric activation)
- **SELU networks**: Use Init.LeCun (self-normalizing property)
- **Experimental/debugging**: Use Init.Random (not recommended for production)
"""

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

    Extends :class:`Layer` with ``get_parameters``, which returns
    ``[(weight, weight_grad), (bias, bias_grad)]`` after a backward pass.
    """

    def get_parameters(self) -> list[ParametersRef]: ...


@runtime_checkable
class FusedLayer(Layer, Protocol):
    """Structural protocol for activation layers that support fused loss gradients.

    ``Sigmoid`` and ``Softmax`` expose ``backward_optimized``, which skips
    the full Jacobian and directly returns the optimised gradient
    ``activation(z) - y_true`` when combined with BCE or CCE loss.
    """

    def backward_optimized(self, output_gradient: Tensor) -> Tensor: ...
