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


# ============================================================================
# LOSS FUNCTION DOCSTRINGS
# ============================================================================

MSE.__doc__ = """
Mean Squared Error loss function.

**Formula:**
    L = (1/n) Σᵢ (yᵢ - ŷᵢ)²

**Gradient (w.r.t. predictions):**
    ∂L/∂ŷ = 2(ŷ - y) / n

**Constructor:**
    ```python
    MSE()
    ```

**Parameters:**
    None (no configuration required).

**Methods:**
    - `call(y_true: Tensor, y_pred: Tensor) -> float`: Compute loss value (scalar)
    - `backward(y_true: Tensor, y_pred: Tensor) -> Tensor`: Compute gradient w.r.t. predictions

**Example:**
    ```python
    from neomatrix import losses
    from neomatrix._backend import Tensor
    
    loss_fn = losses.MSE()
    
    # Forward pass
    y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]]))
    y_pred = Tensor.from_numpy(np.array([[1.1, 1.9, 3.2]]))
    loss_val = loss_fn.call(y_true, y_pred)  # Returns scalar
    
    # Backward pass
    grad = loss_fn.backward(y_true, y_pred)  # Shape: [1, 3]
    ```

**Use Cases:**
    - Regression tasks (continuous output)
    - Metric: penalizes large errors quadratically
    - Sensitive to outliers (use MAE or Huber for robustness)

**Properties:**
    - Differentiable everywhere
    - Convex optimization landscape
    - Range: [0, ∞)
"""

MAE.__doc__ = """
Mean Absolute Error loss function.

**Formula:**
    L = (1/n) Σᵢ |yᵢ - ŷᵢ|

**Gradient (w.r.t. predictions):**
    ∂L/∂ŷ = -sign(y - ŷ) / n

**Constructor:**
    ```python
    MAE()
    ```

**Parameters:**
    None (no configuration required).

**Methods:**
    - `call(y_true: Tensor, y_pred: Tensor) -> float`: Compute loss value (scalar)
    - `backward(y_true: Tensor, y_pred: Tensor) -> Tensor`: Compute gradient w.r.t. predictions

**Example:**
    ```python
    from neomatrix import losses
    from neomatrix._backend import Tensor
    
    loss_fn = losses.MAE()
    
    # Forward pass
    y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]]))
    y_pred = Tensor.from_numpy(np.array([[1.5, 1.8, 3.5]]))
    loss_val = loss_fn.call(y_true, y_pred)  # Returns scalar
    
    # Backward pass
    grad = loss_fn.backward(y_true, y_pred)  # Shape: [1, 3]
    ```

**Use Cases:**
    - Robust regression (less sensitive to outliers than MSE)
    - Metric: penalizes errors linearly
    - Preferred when large errors should not dominate training

**Properties:**
    - Differentiable everywhere except at zero (gradient uses sign function)
    - Convex optimization landscape
    - Range: [0, ∞)
    - More robust to outliers than MSE
"""

BCE.__doc__ = """
Binary Cross-Entropy loss function for binary classification.

**Formula:**
    L = -(1/n) Σᵢ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]

**Gradient (w.r.t. predictions):**
    ∂L/∂ŷ = (ŷ - y) / (ŷ·(1-ŷ)·n)

**Optimized Gradient (fused with Sigmoid activation):**
    When used with Sigmoid output layer:
    ∂L/∂z = σ(z) - y  (where z = pre-activation logits)

**Constructor:**
    ```python
    BCE()
    ```

**Parameters:**
    None (no configuration required).

**Methods:**
    - `call(y_true: Tensor, y_pred: Tensor) -> float`: Compute loss value (scalar)
    - `backward(y_true: Tensor, y_pred: Tensor) -> Tensor`: Compute gradient w.r.t. predictions
    - `backward_optimized(y_true: Tensor, y_pred: Tensor) -> Tensor`: Compute optimized gradient (for Sigmoid + BCE)

**Example:**
    ```python
    from neomatrix import losses
    from neomatrix._backend import Tensor
    
    loss_fn = losses.BCE()
    
    # Standard usage (predictions already passed through Sigmoid)
    y_true = Tensor.from_numpy(np.array([[0.0, 1.0, 1.0]]))  # Binary labels
    y_pred = Tensor.from_numpy(np.array([[0.1, 0.9, 0.8]]))  # Sigmoid outputs
    loss_val = loss_fn.call(y_true, y_pred)
    grad = loss_fn.backward(y_true, y_pred)
    
    # Optimized usage (fused Sigmoid + BCE gradient)
    if isinstance(loss_fn, losses.FusedLossFunction):
        # y_pred here should be PRE-ACTIVATION logits (before Sigmoid)
        optimized_grad = loss_fn.backward_optimized(y_true, y_pred)
    ```

**Use Cases:**
    - Binary classification (two classes: 0 or 1)
    - Multi-label classification (independent binary decisions per label)
    - Output layer: Sigmoid activation

**Properties:**
    - Expects predictions in range (0, 1) — use Sigmoid activation
    - Expects labels to be 0 or 1
    - Numerically stable implementation (log(0) handled internally)
    - Fused gradient optimization available with Sigmoid

**Limitations:**
    - Sensitive to class imbalance (consider weighted BCE or focal loss)
    - Predictions exactly 0 or 1 cause numerical issues (use label smoothing)
"""

CCE.__doc__ = """
Categorical Cross-Entropy loss function for multi-class classification.

**Formula:**
    L = -(1/n) Σᵢ Σₖ yᵢₖ·log(ŷᵢₖ)

**Gradient (w.r.t. predictions):**
    ∂L/∂ŷ = -y / (ŷ·n)

**Optimized Gradient (fused with Softmax activation):**
    When used with Softmax output layer:
    ∂L/∂z = softmax(z) - y  (where z = pre-activation logits)

**Constructor:**
    ```python
    CCE()
    ```

**Parameters:**
    None (no configuration required).

**Methods:**
    - `call(y_true: Tensor, y_pred: Tensor) -> float`: Compute loss value (scalar)
    - `backward(y_true: Tensor, y_pred: Tensor) -> Tensor`: Compute gradient w.r.t. predictions
    - `backward_optimized(y_true: Tensor, y_pred: Tensor) -> Tensor`: Compute optimized gradient (for Softmax + CCE)

**Example:**
    ```python
    from neomatrix import losses
    from neomatrix._backend import Tensor
    import numpy as np
    
    loss_fn = losses.CCE()
    
    # Standard usage (predictions already passed through Softmax)
    y_true = Tensor.from_numpy(np.array([[1.0, 0.0, 0.0]]))  # One-hot encoded
    y_pred = Tensor.from_numpy(np.array([[0.8, 0.15, 0.05]]))  # Softmax outputs
    loss_val = loss_fn.call(y_true, y_pred)
    grad = loss_fn.backward(y_true, y_pred)
    
    # Optimized usage (fused Softmax + CCE gradient)
    if isinstance(loss_fn, losses.FusedLossFunction):
        # y_pred here should be PRE-ACTIVATION logits (before Softmax)
        optimized_grad = loss_fn.backward_optimized(y_true, y_pred)
    ```

**Use Cases:**
    - Multi-class classification (K mutually exclusive classes)
    - Output layer: Softmax activation
    - Single-label classification (each sample belongs to exactly one class)

**Properties:**
    - Expects predictions to sum to 1 per sample (use Softmax)
    - Expects labels to be one-hot encoded (or probability distributions)
    - Numerically stable implementation (log(0) handled internally)
    - Fused gradient optimization available with Softmax

**Limitations:**
    - Not suitable for multi-label classification (use BCE instead)
    - Sensitive to class imbalance (consider weighted CCE or focal loss)
    - Predictions exactly 0 cause numerical issues (use label smoothing)
"""

HuberLoss.__doc__ = """
Huber loss function for robust regression.

**Formula:**
    For |error| ≤ δ (quadratic region):
        L = (1/n) Σᵢ (1/2)·(yᵢ - ŷᵢ)²
    
    For |error| > δ (linear region):
        L = (1/n) Σᵢ δ·(|yᵢ - ŷᵢ| - δ/2)

**Gradient (w.r.t. predictions):**
    ∂L/∂ŷ = {
        (ŷ - y) / n,              if |y - ŷ| ≤ δ
        δ·sign(ŷ - y) / n,        if |y - ŷ| > δ
    }

**Constructor:**
    ```python
    HuberLoss(delta: float = 1.0)
    ```

**Parameters:**
    - `delta` (float): Threshold separating quadratic and linear regions. Default: 1.0.
        - Small δ → more robust to outliers (behaves like MAE)
        - Large δ → less robust, behaves like MSE

**Methods:**
    - `call(y_true: Tensor, y_pred: Tensor) -> float`: Compute loss value (scalar)
    - `backward(y_true: Tensor, y_pred: Tensor) -> Tensor`: Compute gradient w.r.t. predictions

**Example:**
    ```python
    from neomatrix import losses
    from neomatrix._backend import Tensor
    import numpy as np
    
    # Default delta = 1.0
    loss_fn = losses.HuberLoss()
    
    # Custom delta for higher robustness
    robust_loss = losses.HuberLoss(delta=0.5)
    
    # Forward pass
    y_true = Tensor.from_numpy(np.array([[1.0, 2.0, 10.0]]))  # Note: outlier at 10.0
    y_pred = Tensor.from_numpy(np.array([[1.1, 1.9, 3.0]]))
    loss_val = loss_fn.call(y_true, y_pred)
    
    # Backward pass
    grad = loss_fn.backward(y_true, y_pred)
    ```

**Use Cases:**
    - Regression with outliers (robust alternative to MSE)
    - Time series forecasting with noisy data
    - Balancing sensitivity to small vs large errors

**Properties:**
    - Differentiable everywhere
    - Convex optimization landscape
    - Quadratic near zero (like MSE) → faster convergence for small errors
    - Linear for large errors (like MAE) → robust to outliers
    - Range: [0, ∞)

**Tuning Delta:**
    - δ = 0.1: Very robust, almost pure MAE
    - δ = 1.0: Standard Huber loss (default)
    - δ = 10.0: Less robust, closer to MSE
"""

HingeLoss.__doc__ = """
Hinge loss function for SVM-style binary classification.

**Formula:**
    L = (1/n) Σᵢ max(0, 1 - yᵢ·ŷᵢ)

**Gradient (w.r.t. predictions):**
    ∂L/∂ŷ = {
        -y / n,    if y·ŷ < 1 (margin violation)
        0,         if y·ŷ ≥ 1 (correct classification with margin)
    }

**Constructor:**
    ```python
    HingeLoss()
    ```

**Parameters:**
    None (no configuration required).

**Methods:**
    - `call(y_true: Tensor, y_pred: Tensor) -> float`: Compute loss value (scalar)
    - `backward(y_true: Tensor, y_pred: Tensor) -> Tensor`: Compute gradient w.r.t. predictions

**Example:**
    ```python
    from neomatrix import losses
    from neomatrix._backend import Tensor
    import numpy as np
    
    loss_fn = losses.HingeLoss()
    
    # Labels MUST be -1 or +1 (NOT 0 or 1!)
    y_true = Tensor.from_numpy(np.array([[-1.0, 1.0, 1.0]]))
    
    # Raw predictions (no activation needed)
    y_pred = Tensor.from_numpy(np.array([[-0.5, 2.0, 0.3]]))
    
    # Forward pass
    loss_val = loss_fn.call(y_true, y_pred)  # Penalizes first and third sample
    
    # Backward pass
    grad = loss_fn.backward(y_true, y_pred)
    ```

**Use Cases:**
    - Binary classification with maximum-margin objective
    - Support Vector Machine (SVM) training
    - When you want predictions to have confidence margin (not just correct sign)

**Properties:**
    - **CRITICAL**: Expects labels in {-1, +1}, NOT {0, 1}
    - No activation function needed on output (raw scores)
    - Zero gradient when prediction is correct with margin ≥ 1
    - Non-zero gradient only for margin violations
    - Encourages confident predictions (not just correct)
    - Range: [0, ∞)

**Limitations:**
    - Only suitable for binary classification
    - Does not produce probability estimates (use BCE + Sigmoid for that)
    - Sensitive to outliers (no saturation like cross-entropy)
    - Requires label conversion: 0 → -1, 1 → +1

**Comparison with BCE:**
    - Hinge: Stops penalizing once margin ≥ 1 (sparse gradients)
    - BCE: Always penalizes, pushing predictions to 0 or 1 (dense gradients)
"""
