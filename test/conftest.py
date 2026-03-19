"""
Pytest configuration and fixtures for NeoMatrix test suite.

Fixtures use ALL Tensor constructors (from_numpy, zeros, random, direct)
to ensure comprehensive coverage of the Rust backend.
"""

import numpy as np
import pytest

from neomatrix._backend import Tensor
from neomatrix import layers, losses, optimizers, metrics


# ============================================================================
# Tensor Fixtures — from_numpy constructor
# ============================================================================


@pytest.fixture
def tensor_1d():
    """1D tensor [5] created via from_numpy."""
    return Tensor.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))


@pytest.fixture
def tensor_2d():
    """2D tensor [3, 4] created via from_numpy."""
    return Tensor.from_numpy(np.arange(12.0).reshape(3, 4).astype(np.float32))


@pytest.fixture
def tensor_batch():
    """Batch tensor [32, 10] created via from_numpy (32 samples, 10 features)."""
    return Tensor.from_numpy(np.random.randn(32, 10).astype(np.float32))


@pytest.fixture
def tensor_batch_small():
    """Small batch tensor [4, 3] created via from_numpy."""
    return Tensor.from_numpy(
        np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=np.float32,
        )
    )


# ============================================================================
# Tensor Fixtures — direct constructor (Tensor(shape, content))
# ============================================================================


@pytest.fixture
def tensor_direct_1d():
    """1D tensor [4] created via Tensor(shape, content)."""
    return Tensor([4], [10.0, 20.0, 30.0, 40.0])


@pytest.fixture
def tensor_direct_2d():
    """2D tensor [2, 3] created via Tensor(shape, content)."""
    return Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


# ============================================================================
# Tensor Fixtures — zeros constructor
# ============================================================================


@pytest.fixture
def tensor_zeros_1d():
    """1D zero tensor [5] created via Tensor.zeros."""
    return Tensor.zeros([5])


@pytest.fixture
def tensor_zeros_2d():
    """2D zero tensor [3, 4] created via Tensor.zeros."""
    return Tensor.zeros([3, 4])


# ============================================================================
# Tensor Fixtures — random constructor
# ============================================================================


@pytest.fixture
def tensor_random_1d():
    """1D random tensor [5] created via Tensor.random."""
    return Tensor.random([5])


@pytest.fixture
def tensor_random_2d():
    """2D random tensor [3, 4] created via Tensor.random."""
    return Tensor.random([3, 4])


# ============================================================================
# Label Fixtures
# ============================================================================


@pytest.fixture
def labels_onehot():
    """One-hot encoded labels [32, 10] for classification."""
    labels = np.random.randint(0, 10, 32)
    return Tensor.from_numpy(np.eye(10)[labels].astype(np.float32))


@pytest.fixture
def labels_onehot_small():
    """Small one-hot encoded labels [4, 3]."""
    return Tensor.from_numpy(
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    )


@pytest.fixture
def binary_labels():
    """Binary labels [32, 1]."""
    return Tensor.from_numpy(
        np.random.randint(0, 2, 32).reshape(-1, 1).astype(np.float32)
    )


@pytest.fixture
def regression_targets():
    """Regression targets [32, 1]."""
    return Tensor.from_numpy(np.random.randn(32, 1).astype(np.float32))


# ============================================================================
# Layer Fixtures
# ============================================================================


@pytest.fixture
def dense_layer():
    """Dense layer: 10 inputs -> 5 outputs, He init."""
    return layers.Dense(10, 5, layers.Init.He)


@pytest.fixture
def dense_layer_small():
    """Dense layer: 2 inputs -> 3 outputs, Random init."""
    return layers.Dense(2, 3, layers.Init.Random)


@pytest.fixture
def relu_activation():
    """ReLU activation fixture."""
    return layers.ReLU()


@pytest.fixture
def sigmoid_activation():
    """Sigmoid activation fixture."""
    return layers.Sigmoid()


@pytest.fixture
def tanh_activation():
    """Tanh activation fixture."""
    return layers.Tanh()


@pytest.fixture
def softmax_activation():
    """Softmax activation fixture."""
    return layers.Softmax()


# ============================================================================
# Loss Function Fixtures
# ============================================================================


@pytest.fixture
def mse_loss():
    """MSE loss function fixture."""
    return losses.MSE()


@pytest.fixture
def mae_loss():
    """MAE loss function fixture."""
    return losses.MAE()


@pytest.fixture
def bce_loss():
    """BCE loss function fixture."""
    return losses.BCE()


@pytest.fixture
def cce_loss():
    """CCE loss function fixture."""
    return losses.CCE()


@pytest.fixture
def huber_loss():
    """Huber loss function fixture (delta=1.0)."""
    return losses.HuberLoss(1.0)


@pytest.fixture
def hinge_loss():
    """Hinge loss function fixture."""
    return losses.HingeLoss()


# ============================================================================
# Optimizer Fixtures
# ============================================================================


@pytest.fixture
def optimizer():
    """GradientDescent optimizer with lr=0.01."""
    return optimizers.GradientDescent(learning_rate=0.01)


@pytest.fixture
def momentum_optimizer():
    """MomentumGD optimizer with lr=0.01 and momentum=0.9."""
    return optimizers.MomentumGD(learning_rate=0.01, momentum=0.9)


@pytest.fixture
def adagrad_optimizer():
    """Adagrad optimizer with lr=0.01."""
    return optimizers.Adagrad(learning_rate=0.01)


# ============================================================================
# Metric Fixtures
# ============================================================================


@pytest.fixture
def accuracy_metric():
    """Accuracy metric fixture."""
    return metrics.Accuracy()


@pytest.fixture
def precision_metric():
    """Precision metric fixture."""
    return metrics.Precision()


@pytest.fixture
def recall_metric():
    """Recall metric fixture."""
    return metrics.Recall()


@pytest.fixture
def f1_metric():
    """F1Score metric fixture."""
    return metrics.F1Score()


@pytest.fixture
def mse_metric():
    """MSE metric fixture."""
    return metrics.MSE()


@pytest.fixture
def mae_metric():
    """MAE metric fixture."""
    return metrics.MAE()


# ============================================================================
# Assertion Helpers
# ============================================================================


def assert_tensor_shape(tensor: Tensor, expected_shape: list):
    """Assert tensor has expected shape.

    Note: Tensor.shape returns a list, NOT a tuple.
    """
    assert tensor.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {tensor.shape}"
    )


def assert_tensor_in_range(tensor: Tensor, min_val: float, max_val: float):
    """Assert all tensor values are in range [min_val, max_val]."""
    data = tensor.to_numpy()
    assert np.all(data >= min_val) and np.all(data <= max_val), (
        f"Tensor values not in range [{min_val}, {max_val}]"
    )


def assert_gradients_valid(tensor: Tensor):
    """Assert gradients are valid (no NaN, no Inf)."""
    data = tensor.to_numpy()
    assert not np.any(np.isnan(data)), "Gradient contains NaN"
    assert not np.any(np.isinf(data)), "Gradient contains Inf"
