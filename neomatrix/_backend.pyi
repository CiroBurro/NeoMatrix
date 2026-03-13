"""
Type stubs for neomatrix._backend (PyO3 bindings).

This file provides type hints for all Rust types exposed to Python via PyO3.
It enables IDE autocomplete, type checking, and inline documentation.

Generated from: neomatrix-python-wrapper/src/
"""

from typing import Any, Iterator, Optional, Union, List, Tuple, overload
import numpy as np
import numpy.typing as npt

# ============================================================================
# TENSOR
# ============================================================================

class Tensor:
    """
    Multi-dimensional array with NumPy interoperability.

    Implements the NumPy array protocol (__array__) for seamless conversion.
    All operations acquire a mutex lock for thread safety.

    Properties:
        data (np.ndarray): Raw NumPy array view (get/set)
        shape (tuple[int, ...]): Dimensions as tuple (read-only)
        ndim (int): Number of dimensions (read-only)
        dimension (int): Alias for ndim (read-only)

    Supports:
        - Arithmetic: +, -, *, / (element-wise or broadcasting)
        - Indexing: tensor[i, j], tensor[i, j] = value
        - Iteration: for x in tensor (flattened)
        - NumPy: np.array(tensor), tensor.to_numpy()
    """

    def __init__(self, shape: List[int], content: List[float]) -> None:
        """
        Create a tensor with given shape and data.

        Args:
            shape: Dimensions [d1, d2, ...]
            content: Flattened data (length must equal product of shape)

        Raises:
            RuntimeError: If len(content) != prod(shape)

        Example:
            >>> t = Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            >>> t.shape
            (2, 3)
        """
        ...

    @staticmethod
    def zeros(shape: List[int]) -> "Tensor":
        """Create a tensor filled with zeros."""
        ...

    @staticmethod
    def random(shape: List[int], start: float = -1.0, end: float = 1.0) -> "Tensor":
        """Create a tensor with uniform random values in [start, end)."""
        ...

    @staticmethod
    def from_numpy(array: npt.NDArray[np.float32]) -> "Tensor":
        """Create a tensor from a NumPy array (copies data)."""
        ...

    @staticmethod
    def cat(tensors: List["Tensor"], axis: int) -> "Tensor":
        """Concatenate tensors along the given axis."""
        ...

    @property
    def data(self) -> npt.NDArray[np.float32]:
        """Get tensor data as NumPy array (zero-copy when possible)."""
        ...

    @data.setter
    def data(self, value: npt.NDArray[np.float32]) -> None:
        """Set tensor data from NumPy array (copies data)."""
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor dimensions as tuple."""
        ...

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        ...

    @property
    def dimension(self) -> int:
        """Get number of dimensions (alias for ndim)."""
        ...

    def to_numpy(self) -> npt.NDArray[np.float32]:
        """Convert to NumPy array (zero-copy when possible)."""
        ...

    def __array__(self) -> npt.NDArray[np.float32]:
        """NumPy array protocol implementation (enables np.array(tensor))."""
        ...

    def length(self) -> int:
        """Total number of elements (product of shape)."""
        ...

    def dot(self, other: "Tensor") -> "Tensor":
        """
        Matrix multiplication / dot product.

        Supports:
            - 1D × 1D → scalar (shape [])
            - 1D × 2D → 1D
            - 2D × 1D → 1D
            - 2D × 2D → 2D

        Raises:
            RuntimeError: If dimensions are incompatible
        """
        ...

    def transpose(self) -> "Tensor":
        """Return transposed copy (2D only)."""
        ...

    def transpose_inplace(self) -> None:
        """Transpose in-place (2D only)."""
        ...

    def reshape(self, shape: List[int]) -> "Tensor":
        """Return reshaped copy."""
        ...

    def reshape_inplace(self, shape: List[int]) -> None:
        """Reshape in-place."""
        ...

    def flatten(self) -> "Tensor":
        """Return flattened copy (1D)."""
        ...

    def flatten_inplace(self) -> None:
        """Flatten in-place (1D)."""
        ...

    def push(self, tensor: "Tensor", axis: int) -> None:
        """Concatenate tensor along axis (mutates self)."""
        ...

    def push_row(self, tensor: "Tensor") -> None:
        """Append row (mutates self, promotes 1D→2D if needed)."""
        ...

    def push_column(self, tensor: "Tensor") -> None:
        """Append column (mutates self, promotes 1D→2D if needed)."""
        ...

    def cat_inplace(self, tensors: List["Tensor"], axis: int) -> None:
        """Concatenate multiple tensors along axis (mutates self)."""
        ...

    # Arithmetic operators
    def __add__(self, other: Union["Tensor", float]) -> "Tensor": ...
    def __radd__(self, other: float) -> "Tensor": ...
    def __sub__(self, other: Union["Tensor", float]) -> "Tensor": ...
    def __rsub__(self, other: float) -> "Tensor": ...
    def __mul__(self, other: Union["Tensor", float]) -> "Tensor": ...
    def __rmul__(self, other: float) -> "Tensor": ...
    def __truediv__(self, other: Union["Tensor", float]) -> "Tensor": ...
    def __rtruediv__(self, other: float) -> "Tensor": ...
    def __neg__(self) -> "Tensor": ...

    # Indexing
    @overload
    def __getitem__(self, index: int) -> float: ...
    @overload
    def __getitem__(self, index: Tuple[int, ...]) -> float: ...
    @overload
    def __setitem__(self, index: int, value: float) -> None: ...
    @overload
    def __setitem__(self, index: Tuple[int, ...], value: float) -> None: ...
    def __len__(self) -> int:
        """Total number of elements."""
        ...

    def __iter__(self) -> Iterator[float]:
        """Iterate over flattened elements."""
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# LAYERS
# ============================================================================

class Dense:
    """
    Fully-connected neural network layer: Y = X·W + b

    Parameters:
        - Weights (W): shape (in_features, out_features)
        - Biases (b): shape (out_features,)
        - Weight gradients (∇W): accumulated during backprop
        - Bias gradients (∇b): accumulated during backprop

    All parameters are shared via Arc<Mutex<Tensor>> with the optimizer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        init: Optional["Init"] = None,
        range_start: Optional[float] = None,
        range_end: Optional[float] = None,
    ) -> None:
        """
        Create a Dense layer.

        Args:
            input_size: Number of input features (fan-in)
            output_size: Number of output features (fan-out)
            init: Weight initialization strategy (Xavier/He/LeCun/Random). Defaults to Xavier.
            range_start: Start of uniform range (only for Init.Random)
            range_end: End of uniform range (only for Init.Random)

        Example:
            >>> layer = Dense(784, 128, init=Init.He)
            >>> output = layer.forward(input, training=True)
        """
        ...

    def forward(self, input: Tensor, training: bool) -> Tensor:
        """
        Forward pass: Y = X·W + b

        Args:
            input: Input tensor (batch_size, input_size)
            training: Whether in training mode (stores input for backprop)

        Returns:
            Output tensor (batch_size, output_size)
        """
        ...

    def backward(self, output_gradient: Tensor) -> Tensor:
        """
        Backward pass: compute gradients and return input gradient.

        Accumulates weight/bias gradients internally (accessed by optimizer).

        Args:
            output_gradient: Gradient from next layer (batch_size, out_features)

        Returns:
            Input gradient (batch_size, in_features)

        Raises:
            RuntimeError: If forward() was not called first
        """
        ...

    def get_parameters(self) -> "ParametersRef":
        """
        Get shared references to layer parameters for optimizer registration.

        Returns:
            ParametersRef with Arc<Mutex<Tensor>> for weights/biases/gradients
        """
        ...

class ReLU:
    """ReLU activation layer: f(x) = max(0, x)"""

    def __init__(self) -> None: ...
    def forward(self, input: Tensor, training: bool) -> Tensor:
        """Apply ReLU activation element-wise."""
        ...

    def backward(self, output_gradient: Tensor) -> Tensor:
        """
        Backprop through ReLU: gradient is 1 where x > 0, else 0.

        Args:
            output_gradient: Gradient from next layer

        Returns:
            Input gradient (element-wise multiplication)
        """
        ...

class Sigmoid:
    """Sigmoid activation layer: f(x) = 1 / (1 + e^(-x))"""

    def __init__(self) -> None: ...
    def forward(self, input: Tensor, training: bool) -> Tensor:
        """Apply sigmoid activation element-wise."""
        ...

    def backward(self, output_gradient: Tensor) -> Tensor:
        """
        Backprop through sigmoid: gradient is σ(x) · (1 - σ(x)).

        This computes the classical derivative (general-purpose).
        """
        ...

    def backward_optimized(self, output_gradient: Tensor) -> Tensor:
        """
        Optimized gradient passthrough for Sigmoid + BCE fusion.

        When using BCE.backward_optimized(), the gradient is already
        simplified to (σ(z) - y). This method passes it through unchanged.
        """
        ...

class Tanh:
    """Tanh activation layer: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))"""

    def __init__(self) -> None: ...
    def forward(self, input: Tensor, training: bool) -> Tensor:
        """Apply tanh activation element-wise."""
        ...

    def backward(self, output_gradient: Tensor) -> Tensor:
        """
        Backprop through tanh: gradient is 1 - tanh²(x).

        This computes the classical derivative (general-purpose).
        """
        ...

class Softmax:
    """Softmax activation layer: f(x_i) = e^(x_i) / Σ_j e^(x_j)"""

    def __init__(self) -> None: ...
    def forward(self, input: Tensor, training: bool) -> Tensor:
        """
        Apply softmax activation (converts logits to probabilities).

        Uses log-sum-exp trick for numerical stability: x - max(x) before exp.
        """
        ...

    def backward(self, output_gradient: Tensor) -> Tensor:
        """
        Backprop through softmax: computes Jacobian matrix.

        This is the classical derivative (general-purpose, works with any loss).
        """
        ...

    def backward_optimized(self, output_gradient: Tensor) -> Tensor:
        """
        Optimized gradient passthrough for Softmax + CCE fusion.

        When using CCE.backward_optimized(), the gradient is already
        simplified to (softmax(z) - y). This method passes it through
        without Jacobian computation.
        """
        ...

# ============================================================================
# INITIALIZATION
# ============================================================================

class Init:
    """
    Weight initialization strategies for neural network layers.

    Variants:
        - Xavier: N(0, √(2/(n_in + n_out))) — best for Sigmoid/Tanh
        - He: N(0, √(2/n_in)) — best for ReLU
        - LeCun: N(0, √(1/n_in)) — best for SELU
        - Random: U(a, b) — legacy, not recommended
    """

    Xavier: "Init"
    He: "Init"
    LeCun: "Init"
    Random: "Init"

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class MSE:
    """
    Mean Squared Error loss: L = (1/n) Σ(y - ŷ)²

    Best for: Regression tasks
    """

    def __init__(self) -> None: ...
    def call(self, y_true: Tensor, y_pred: Tensor) -> float:
        """
        Compute MSE loss.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Scalar loss value
        """
        ...

    def backward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute gradient: ∇L/∂ŷ = (2/n)(ŷ - y)

        Returns:
            Gradient tensor (same shape as y_pred)
        """
        ...

class MAE:
    """
    Mean Absolute Error loss: L = (1/n) Σ|y - ŷ|

    Best for: Robust regression (less sensitive to outliers than MSE)
    """

    def __init__(self) -> None: ...
    def call(self, y_true: Tensor, y_pred: Tensor) -> float:
        """Compute MAE loss."""
        ...

    def backward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute gradient: ∇L/∂ŷ = (1/n) · sign(ŷ - y)

        Returns:
            Gradient tensor (same shape as y_pred)
        """
        ...

class BCE:
    """
    Binary Cross-Entropy loss: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

    Best for: Binary classification (with Sigmoid activation)
    """

    def __init__(self) -> None: ...
    def call(self, y_true: Tensor, y_pred: Tensor) -> float:
        """
        Compute BCE loss.

        Adds epsilon (1e-15) for numerical stability.

        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted probabilities (0 to 1)

        Returns:
            Scalar loss value
        """
        ...

    def backward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute gradient (classical): ∇L/∂ŷ = (ŷ - y) / [ŷ(1-ŷ)]

        Use this with any activation function (not just Sigmoid).
        """
        ...

    def backward_optimized(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Optimized gradient for Sigmoid + BCE fusion: ∇L/∂z = σ(z) - y

        Use this when the last layer is Sigmoid. Requires calling
        Sigmoid.backward_optimized() instead of Sigmoid.backward().

        Returns:
            Simplified gradient (same shape as y_pred)
        """
        ...

class CCE:
    """
    Categorical Cross-Entropy loss: L = -Σ y_i · log(ŷ_i)

    Best for: Multi-class classification (with Softmax activation)
    """

    def __init__(self) -> None: ...
    def call(self, y_true: Tensor, y_pred: Tensor) -> float:
        """
        Compute CCE loss.

        Adds epsilon (1e-15) for numerical stability.

        Args:
            y_true: One-hot encoded labels (shape: [batch, num_classes])
            y_pred: Predicted probabilities (shape: [batch, num_classes])

        Returns:
            Scalar loss value
        """
        ...

    def backward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute gradient (classical): ∇L/∂ŷ = -y / ŷ

        Use this with any activation function (not just Softmax).
        """
        ...

    def backward_optimized(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Optimized gradient for Softmax + CCE fusion: ∇L/∂z = softmax(z) - y

        Use this when the last layer is Softmax. Requires calling
        Softmax.backward_optimized() instead of Softmax.backward().

        Returns:
            Simplified gradient (same shape as y_pred)
        """
        ...

class HuberLoss:
    """
    Huber loss: Smooth transition between MSE and MAE.

    L = { 0.5(y - ŷ)²         if |y - ŷ| ≤ δ
        { δ(|y - ŷ| - 0.5δ)   otherwise

    Best for: Robust regression (less sensitive to outliers than MSE)
    """

    def __init__(self, delta: float) -> None:
        """
        Create Huber loss.

        Args:
            delta: Threshold for switching from MSE to MAE
        """
        ...

    def call(self, y_true: Tensor, y_pred: Tensor) -> float:
        """Compute Huber loss."""
        ...

    def backward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute gradient:
            ∇L/∂ŷ = { ŷ - y           if |y - ŷ| ≤ δ
                    { δ · sign(ŷ - y)  otherwise
        """
        ...

class HingeLoss:
    """
    Hinge loss: L = max(0, 1 - y·ŷ)

    Best for: SVM-style binary classification (y ∈ {-1, +1})
    """

    def __init__(self) -> None: ...
    def call(self, y_true: Tensor, y_pred: Tensor) -> float:
        """
        Compute hinge loss.

        Args:
            y_true: Ground truth labels (-1 or +1)
            y_pred: Raw prediction scores (not probabilities)

        Returns:
            Scalar loss value
        """
        ...

    def backward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute gradient:
            ∇L/∂ŷ = { -y  if y·ŷ < 1
                    { 0   otherwise
        """
        ...

# ============================================================================
# OPTIMIZERS
# ============================================================================

class GradientDescent:
    """
    Stochastic Gradient Descent optimizer: θ = θ - lr·∇θ

    Implements stateful parameter updates with PyTorch-style API.
    Parameters are registered once, then updated repeatedly via step().
    All updates are parallelized using Rayon (~2-4x speedup on multi-layer networks).

    Usage:
        optimizer = GradientDescent(learning_rate=0.01)
        params = [layer.get_parameters() for layer in layers]
        optimizer.register_params(params)

        for batch in batches:
            optimizer.zero_grad()
            loss = forward_and_compute_loss(batch)
            backward_pass()
            optimizer.step()
    """

    def __init__(self, learning_rate: float) -> None:
        """
        Create a GradientDescent optimizer.

        Args:
            learning_rate: Step size for parameter updates
        """
        ...

    def register_params(self, params: List["ParametersRef"]) -> None:
        """
        Register layer parameters for optimization.

        Call this once after creating all layers (typically in Model.compile()).

        Args:
            params: List of ParametersRef from layer.get_parameters()
        """
        ...

    def step(self) -> None:
        """
        Update all registered parameters: θ = θ - lr·∇θ

        Parallelized across layers using Rayon. Typical time: ~4ms for
        10-layer network with 1000→50 neurons per layer.

        Raises:
            RuntimeError: If register_params() was not called first
        """
        ...

    def zero_grad(self) -> None:
        """
        Reset all gradients to zero.

        MUST be called before each forward pass (gradients accumulate by design).
        Parallelized across layers using Rayon.

        Raises:
            RuntimeError: If register_params() was not called first
        """
        ...

class ParametersRef:
    """
    Shared reference container for layer parameters.

    Holds Arc<Mutex<Tensor>> for:
        - weights: Weight matrix
        - biases: Bias vector
        - w_grads: Weight gradients (accumulated during backprop)
        - b_grads: Bias gradients (accumulated during backprop)

    Obtained via layer.get_parameters() and passed to optimizer.register_params().
    The optimizer and layer share ownership of the same tensors via Arc.
    """

    def __init__(
        self,
        weights: Tensor,
        biases: Tensor,
        w_grads: Tensor,
        b_grads: Tensor,
    ) -> None:
        """
        Create a ParametersRef with shared tensor references.

        Args:
            weights: Weight tensor (shared via Arc<Mutex<Tensor>>)
            biases: Bias tensor (shared via Arc<Mutex<Tensor>>)
            w_grads: Weight gradients tensor (shared via Arc<Mutex<Tensor>>)
            b_grads: Bias gradients tensor (shared via Arc<Mutex<Tensor>>)
        """
        ...

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__: str
__all__: List[str]
