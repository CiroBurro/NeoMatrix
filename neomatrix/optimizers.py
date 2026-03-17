"""
Optimizer types exposed by the NeoMatrix Rust backend.

Available: GradientDescent (SGD with configurable learning rate).
"""

from typing import Protocol, runtime_checkable

from neomatrix._backend import GradientDescent, MomentumGD, ParametersRef

__all__ = [
    "GradientDescent",
    "MomentumGD",
    "ParametersRef",
    "Optimizer",
]


@runtime_checkable
class Optimizer(Protocol):
    """Structural protocol for any optimizer.

    All optimizers expose ``register_params()``, ``step()``, and ``zero_grad()``
    for stateful parameter updates during training.
    """

    def register_params(self, params: list[ParametersRef]): ...
    def step(self): ...
    def zero_grad(self): ...


# ============================================================================
# OPTIMIZER DOCSTRINGS
# ============================================================================

GradientDescent.__doc__ = """
Stochastic Gradient Descent (SGD) optimizer.

**Algorithm:**
    For each parameter θ (weights, biases):
        θ ← θ - η·∇θ

    Where:
    - η = learning_rate
    - ∇θ = gradient computed during backpropagation

**Constructor:**
    ```python
    GradientDescent(learning_rate: float)
    ```

**Parameters:**
    - `learning_rate` (float): Step size for weight updates. Typical range: [0.001, 0.1].

**Methods:**
    - `register_params(params: list[ParametersRef])`: Register layer parameters for optimization.
        Must be called BEFORE training (typically in `Model.compile()`).
    - `step()`: Update all registered parameters using accumulated gradients.
        Call AFTER `backward()` to apply weight updates.
    - `zero_grad()`: Reset all gradients to zero.
        Call BEFORE each `forward()` pass (gradients accumulate by design).

**Example:**
    ```python
    from neomatrix import optimizers, layers
    from neomatrix._backend import Tensor
    import numpy as np

    # Create optimizer
    opt = optimizers.GradientDescent(learning_rate=0.01)

    # Create layers
    layer1 = layers.Dense(128, 784, layers.Init.He)
    layer2 = layers.Dense(10, 128, layers.Init.Xavier)

    # Register parameters (REQUIRED before training)
    params = [
        layer1.get_parameters(),
        layer2.get_parameters(),
    ]
    opt.register_params(params)

    # Training loop (single step)
    opt.zero_grad()                              # 1. Reset gradients
    y_pred = layer2.forward(layer1.forward(x, True), True)  # 2. Forward pass
    loss_val = loss_fn.call(y_true, y_pred)      # 3. Compute loss
    grad = loss_fn.backward(y_true, y_pred)      # 4. Loss gradient
    grad = layer2.backward(grad)                 # 5. Backprop layer 2
    grad = layer1.backward(grad)                 # 6. Backprop layer 1
    opt.step()                                   # 7. Update weights
    ```

**Use Cases:**
    - Standard optimizer for most neural network training
    - Works well with momentum-based variants (future: Adam, RMSprop)
    - Baseline for comparing advanced optimizers

**Properties:**
    - **Stateful**: Stores references to all layer parameters via `Arc<Mutex<Tensor>>` (Rust)
    - **Shared Ownership**: Layers and optimizer share the same parameter tensors
    - **Gradient Accumulation**: Multiple `backward()` calls accumulate gradients (reset with `zero_grad()`)
    - **Parallel Updates**: Parameters updated in parallel via Rayon (2-4x speedup on multi-core CPUs)
    - **Thread-Safe**: Mutex locks ensure safe concurrent access to parameters

**Performance:**
    - Average `step()` time: ~4ms (10-layer network, 1000→50 neurons per layer)
    - Average `zero_grad()` time: ~0.7ms
    - Lock overhead: <0.1% of total time (negligible)

**Learning Rate Guidelines:**
    - Too small (< 0.001): Slow convergence, may get stuck in plateaus
    - Too large (> 0.1): Unstable training, divergence, loss oscillations
    - Typical starting point: 0.01 for regression, 0.001 for classification
    - Use learning rate schedules for better convergence (future feature)

**Limitations:**
    - No adaptive learning rates (use Adam for that)
    - No momentum (all updates independent)
    - Sensitive to learning rate choice
    - May struggle with saddle points and ravines

**Advanced Usage (Shared Ownership Pattern):**
    ```python
    # Parameters are shared between layer and optimizer
    dense = layers.Dense(10, 5)
    params_ref = dense.get_parameters()

    # Both point to same memory (Arc<Mutex<Tensor>> in Rust)
    print(params_ref.weights.shape)  # [5, 10]

    opt.register_params([params_ref])

    # Calling dense.backward() modifies params_ref.w_grads
    # Calling opt.step() modifies params_ref.weights
    # No copying — all operations on shared tensors
    ```
"""
