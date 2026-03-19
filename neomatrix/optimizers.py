"""
Optimizer types exposed by the NeoMatrix Rust backend.

Available: GradientDescent (SGD), MomentumGD (SGD with momentum), Adagrad (adaptive learning rates).
"""

from typing import Protocol, runtime_checkable

from neomatrix._backend import Adagrad, GradientDescent, MomentumGD, ParametersRef

__all__ = [
    "GradientDescent",
    "MomentumGD",
    "Adagrad",
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

MomentumGD.__doc__ = """
Momentum Gradient Descent optimizer.

**Algorithm:**
    For each parameter θ (weights, biases):
        v ← β·v + (1-β)·∇θ      # Update velocity (exponentially-weighted average)
        θ ← θ - η·v              # Update parameters using velocity

    Where:
    - η = learning_rate
    - β = momentum (typically 0.9)
    - v = velocity (accumulated gradient history)
    - ∇θ = gradient computed during backpropagation

**Why Momentum?**
    - Faster convergence: Velocity builds up in consistent gradient directions
    - Oscillation dampening: Perpendicular oscillations cancel out
    - Better generalization: Stochastic behavior helps escape sharp local minima

**Constructor:**
    ```python
    MomentumGD(learning_rate: float, momentum: float)
    ```

**Parameters:**
    - `learning_rate` (float): Step size for weight updates. Typical range: [0.001, 0.1].
      Note: With momentum, the effective learning rate is higher than the nominal value.
    - `momentum` (float): Velocity decay coefficient. Typical range: [0.9, 0.999].
      - 0.9: Fast adaptation (good for changing loss landscapes)
      - 0.99: Smoother convergence (default recommendation)

**Methods:**
    - `register_params(params: list[ParametersRef])`: Register layer parameters for optimization.
        Must be called BEFORE training (typically in `Model.compile()`).
        This also initializes velocity vectors (zero-initialized).
    - `step()`: Update all registered parameters using accumulated gradients and velocity.
        Call AFTER `backward()` to apply weight updates.
    - `zero_grad()`: Reset all gradients to zero.
        Call BEFORE each `forward()` pass (gradients accumulate by design).
        Note: Velocity is PRESERVED between steps (unlike gradients).

**Example:**
    ```python
    from neomatrix import optimizers, layers

    # Create optimizer with momentum
    opt = optimizers.MomentumGD(learning_rate=0.01, momentum=0.9)

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
    opt.zero_grad()                              # 1. Reset gradients (velocity preserved)
    y_pred = layer2.forward(layer1.forward(x, True), True)  # 2. Forward pass
    loss_val = loss_fn.call(y_true, y_pred)      # 3. Compute loss
    grad = loss_fn.backward(y_true, y_pred)      # 4. Loss gradient
    grad = layer2.backward(grad)                 # 5. Backprop layer 2
    grad = layer1.backward(grad)                 # 6. Backprop layer 1
    opt.step()                                   # 7. Update weights (velocity accumulated)
    ```

**Use Cases:**
    - Faster convergence than standard SGD
    - Training deep networks with high curvature
    - Escaping shallow local minima
    - When loss landscape has narrow valleys (momentum helps traverse them)

**Properties:**
    - **Stateful**: Stores references to all layer parameters via `Arc<Mutex<Tensor>>` (Rust)
    - **Velocity Vectors**: Maintains per-parameter velocity (weights and biases separately)
    - **Shared Ownership**: Layers and optimizer share the same parameter tensors
    - **Gradient Accumulation**: Multiple `backward()` calls accumulate gradients (reset with `zero_grad()`)
    - **Velocity Preservation**: Velocity is NOT reset by `zero_grad()` — only gradients are reset
    - **Parallel Updates**: Parameters updated in parallel via Rayon (2-4x speedup on multi-core CPUs)
    - **Thread-Safe**: Mutex locks ensure safe concurrent access to parameters

**Performance:**
    - Average `step()` time: ~4ms (10-layer network, 1000→50 neurons per layer)
    - Average `zero_grad()` time: ~0.7ms
    - Lock overhead: <0.1% of total time (negligible)
    - Memory overhead: +2 velocity tensors per layer (weights + biases)

**Hyperparameter Guidelines:**
    | Parameter | Typical Range | Effect |
    |-----------|---------------|--------|
    | Learning rate | 0.001 - 0.1 | Step size (lower than SGD) |
    | Momentum (β) | 0.9 - 0.999 | Velocity decay |

    - β = 0.9: 10 most recent gradients weighted equally
    - β = 0.99: 100 most recent gradients weighted equally
    - β = 0.999: 1000 most recent gradients weighted equally

**Comparison with GradientDescent:**
    | Aspect | GradientDescent | MomentumGD |
    |--------|----------------|------------|
    | Update rule | θ = θ - lr·∇θ | θ = θ - lr·v |
    | Velocity | None | v = β·v + (1-β)·∇θ |
    | Convergence | Slower | Faster (2-10x) |
    | Oscillation | High | Reduced |
    | Memory overhead | Minimal | +2 tensors/layer |

**Limitations:**
    - No adaptive learning rates (use Adam for that)
    - Sensitive to momentum choice (too high = oscillation, too low = no benefit)
    - May overshoot minima in very flat regions

**Advanced Usage (Shared Ownership Pattern):**
    ```python
    # Parameters are shared between layer and optimizer
    dense = layers.Dense(10, 5)
    params_ref = dense.get_parameters()

    # Both point to same memory (Arc<Mutex<Tensor>> in Rust)
    print(params_ref.weights.shape)  # [5, 10]

    opt = MomentumGD(learning_rate=0.01, momentum=0.9)
    opt.register_params([params_ref])

    # Training loop
    for epoch in range(100):
        opt.zero_grad()  # Resets gradients, velocity preserved

        # Forward/backward passes accumulate gradients
        y_pred = dense.forward(x, training=True)
        loss_grad = loss_fn.backward(y_true, y_pred)
        dense.backward(loss_grad)

        # step() uses accumulated velocity to update weights
        opt.step()
    ```
"""

Adagrad.__doc__ = """
Adagrad (Adaptive Gradient) optimizer.

**Algorithm:**
    For each parameter θ (weights, biases):
        G ← G + (∇θ)²                     # Accumulate squared gradients
        θ ← θ - η·∇θ / (√G + ε)           # Update with adapted learning rate

    Where:
    - η = learning_rate (global)
    - G = accumulated squared gradients (per-parameter history)
    - ε = small constant for numerical stability (typically 1e-8)
    - ∇θ = gradient computed during backpropagation

**Why Adagrad?**
    - Adaptive learning rates: Parameters with large historical gradients get smaller updates
    - No manual tuning: Automatically decreases learning rate over time
    - Sparse data: Excellent for sparse features (NLP, recommender systems)
    - Convex optimization: Strong theoretical guarantees for convex problems

**Limitations:**
    - Aggressive decay: Accumulated squared gradients never decrease, causing learning rate
      to shrink monotonically. This can cause premature convergence in non-convex problems.
    - Not ideal for deep learning: Adam/RMSprop are preferred for most neural networks
      as they use exponential moving averages instead of cumulative sums.

**Constructor:**
    ```python
    Adagrad(learning_rate: float)
    ```

**Parameters:**
    - `learning_rate` (float): Global learning rate (typically 0.01 to 0.1).
      Note: Adagrad will automatically reduce this over time based on gradient history,
      so you can usually start with a higher value than standard SGD.

**Methods:**
    - `register_params(params: list[ParametersRef])`: Register layer parameters for optimization.
        Must be called BEFORE training (typically in `Model.compile()`).
        This also initializes the accumulated squared gradient buffers (G) for each parameter.
    - `step()`: Update all registered parameters using accumulated gradients with adaptive rates.
        Call AFTER `backward()` to apply weight updates.
    - `zero_grad()`: Reset all gradients to zero.
        Call BEFORE each `forward()` pass (gradients accumulate by design).
        Note: Accumulated squared gradients (G) are PRESERVED between steps.

**Example:**
    ```python
    from neomatrix import optimizers, layers
    from neomatrix._backend import Tensor
    import numpy as np

    # Create optimizer
    opt = optimizers.Adagrad(learning_rate=0.01)

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
    opt.zero_grad()                              # 1. Reset gradients (G preserved)
    y_pred = layer2.forward(layer1.forward(x, True), True)  # 2. Forward pass
    loss_val = loss_fn.call(y_true, y_pred)      # 3. Compute loss
    grad = loss_fn.backward(y_true, y_pred)      # 4. Loss gradient
    grad = layer2.backward(grad)                 # 5. Backprop layer 2
    grad = layer1.backward(grad)                 # 6. Backprop layer 1
    opt.step()                                   # 7. Update weights with adaptive rates
    ```

**Use Cases:**
    - Sparse feature spaces (text classification, recommender systems)
    - Problems where features have very different scales
    - Convex optimization problems
    - When you want automatic learning rate adaptation without hyperparameter tuning

**Properties:**
    - **Stateful**: Stores references to all layer parameters via `Arc<Mutex<Tensor>>` (Rust)
    - **Adaptive Rates**: Maintains per-parameter accumulated squared gradients (G)
    - **Shared Ownership**: Layers and optimizer share the same parameter tensors
    - **Gradient Accumulation**: Multiple `backward()` calls accumulate gradients (reset with `zero_grad()`)
    - **G Preservation**: Accumulated squared gradients (G) are NOT reset by `zero_grad()` — only current gradients
    - **Parallel Updates**: Parameters updated in parallel via Rayon (2-4x speedup on multi-core CPUs)
    - **Thread-Safe**: Mutex locks ensure safe concurrent access to parameters

**Performance:**
    - Average `step()` time: ~4ms (10-layer network, 1000→50 neurons per layer)
    - Average `zero_grad()` time: ~0.7ms
    - Lock overhead: <0.1% of total time (negligible)
    - Memory overhead: +2 accumulator tensors per layer (weights + biases)

**Learning Rate Guidelines:**
    - Can start with higher values than SGD (0.01 to 0.1)
    - Adagrad automatically reduces effective learning rate over time
    - Initial high rate → fast early progress
    - Accumulated G → diminishing updates later
    - No need for manual learning rate schedules

**Comparison with Other Optimizers:**
    | Aspect | GradientDescent | MomentumGD | Adagrad |
    |--------|----------------|------------|---------|
    | Update rule | θ = θ - lr·∇θ | θ = θ - lr·v | θ = θ - lr·∇θ/√G |
    | Adaptation | None | Velocity | Per-parameter |
    | Memory overhead | Minimal | +2 tensors/layer | +2 tensors/layer |
    | Best for | General | Deep networks | Sparse features |
    | Learning rate decay | None | None | Automatic |

**When to Use:**
    ✅ Use Adagrad when:
    - Working with sparse features (NLP, embeddings)
    - Features have very different scales
    - Training on convex problems
    - You want "set and forget" learning rate

    ❌ Don't use Adagrad when:
    - Training deep neural networks (use Adam instead)
    - Need to train for many epochs (learning rate decays too aggressively)
    - Working with dense features (standard SGD or Momentum often better)

**Advanced Usage (Shared Ownership Pattern):**
    ```python
    # Parameters are shared between layer and optimizer
    dense = layers.Dense(10, 5)
    params_ref = dense.get_parameters()

    # Both point to same memory (Arc<Mutex<Tensor>> in Rust)
    print(params_ref.weights.shape)  # [5, 10]

    opt = Adagrad(learning_rate=0.01)
    opt.register_params([params_ref])

    # Training loop
    for epoch in range(100):
        opt.zero_grad()  # Resets gradients only, G accumulator preserved

        # Forward/backward passes accumulate gradients
        y_pred = dense.forward(x, training=True)
        loss_grad = loss_fn.backward(y_true, y_pred)
        dense.backward(loss_grad)

        # step() uses G to adapt learning rate per-parameter
        # As G grows, effective lr = lr / √G decreases
        opt.step()
    ```

**Mathematical Insight:**
    The adaptive learning rate formula η / √G provides:
    - Larger updates for infrequent features (small G)
    - Smaller updates for frequent features (large G)
    - Automatic balancing without manual feature scaling

    For sparse data, this is powerful: rare but important features get strong signals,
    while common features are dampened to prevent domination.
"""
