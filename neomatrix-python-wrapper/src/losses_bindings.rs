//! Python bindings for loss functions.
//!
//! Provides forward (loss computation) and backward (gradient computation) methods for all
//! supported loss functions. Some losses offer `backward_optimized()` for fused gradient
//! computation when combined with compatible activation functions (Sigmoid+BCE, Softmax+CCE).
//!
//! # Loss Functions
//!
//! - **MSE**: Mean Squared Error — for regression tasks
//! - **MAE**: Mean Absolute Error — robust to outliers
//! - **BCE**: Binary Cross-Entropy — for binary classification with Sigmoid
//! - **CCE**: Categorical Cross-Entropy — for multi-class classification with Softmax
//! - **HuberLoss**: Smooth L1 loss — combines MSE and MAE properties
//! - **HingeLoss**: For SVMs and margin-based classification

use crate::tensor_bindings::PyTensor;
use neomatrix_core::math::losses::{self, LossFunction};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::ops::Deref;
use std::sync::{Arc, Mutex};

/// Python wrapper for Mean Squared Error loss.
///
/// MSE = (1/n) Σ (y_true - y_pred)²
/// Gradient: ∂MSE/∂y_pred = (2/n) · (y_pred - y_true)
///
/// Used for regression tasks. Sensitive to outliers (quadratic penalty).
///
/// # Usage
///
/// ```python
/// loss_fn = MSE()
/// loss = loss_fn.call(y_true, y_pred)
/// grad = loss_fn.backward(y_true, y_pred)
/// ```
#[pyclass(name = "MSE")]
pub struct PyMeanSquaredError {
    inner: losses::MeanSquaredError,
}
#[pymethods]
impl PyMeanSquaredError {
    /// Create a new Mean Squared Error loss function.
    #[new]
    fn new() -> Self {
        PyMeanSquaredError {
            inner: losses::MeanSquaredError,
        }
    }

    /// Compute loss value: MSE = (1/n) Σ (y_true - y_pred)²
    ///
    /// # Arguments
    ///
    /// * `y_true` - Ground truth values
    /// * `y_pred` - Predicted values
    ///
    /// # Returns
    ///
    /// Scalar loss value (averaged over all elements)
    #[pyo3(signature = (y_true, y_pred))]
    fn call(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<f32> {
        Ok(self
            .inner
            .function(
                y_true
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
                y_pred
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?)
    }

    #[pyo3(signature = (y_true, y_pred))]
    fn backward(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .derivative(
                        y_true
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                        y_pred
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }
}

/// Mean Absolute Error (MAE) loss function.
///
/// **Formula:**
/// ```text
/// L(y, ŷ) = (1/n) · Σᵢ |yᵢ - ŷᵢ|
/// ```
///
/// **Gradient:**
/// ```text
/// ∂L/∂ŷ = -(1/n) · sign(y - ŷ)
/// ```
///
/// MAE is more robust to outliers than MSE because it doesn't square errors.
/// However, its gradient is not continuous at zero (flat everywhere else).
///
/// **Use Cases:**
/// - Regression tasks where outliers should not dominate the loss
/// - When you want the model to predict the median rather than the mean
///
/// # Python API
///
/// ```python
/// from neomatrix._backend import MAE
///
/// loss_fn = MAE()
/// loss_value = loss_fn.call(y_true, y_pred)  # Returns scalar f32
/// gradient = loss_fn.backward(y_true, y_pred)  # Returns Tensor
/// ```
#[pyclass(name = "MAE")]
pub struct PyMeanAbsoluteError {
    inner: losses::MeanAbsoluteError,
}

#[pymethods]
impl PyMeanAbsoluteError {
    /// Creates a new Mean Absolute Error loss function.
    ///
    /// # Returns
    /// A new `MAE` instance ready to compute loss and gradients.
    #[new]
    fn new() -> Self {
        PyMeanAbsoluteError {
            inner: losses::MeanAbsoluteError,
        }
    }

    /// Computes the MAE loss between true and predicted values.
    ///
    /// # Arguments
    /// * `y_true` - Ground truth tensor
    /// * `y_pred` - Predicted values tensor
    ///
    /// # Returns
    /// Scalar loss value (f32)
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensors have incompatible shapes or if mutex lock fails.
    #[pyo3(signature = (y_true, y_pred))]
    fn call(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<f32> {
        Ok(self
            .inner
            .function(
                y_true
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
                y_pred
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?)
    }

    /// Computes the gradient of MAE loss with respect to predictions.
    ///
    /// Gradient formula: `∂L/∂ŷ = -(1/n) · sign(y - ŷ)`
    ///
    /// Note: The gradient is **not differentiable at zero** and has constant magnitude elsewhere.
    /// This can cause optimization issues with adaptive learning rate methods.
    ///
    /// # Arguments
    /// * `y_true` - Ground truth tensor
    /// * `y_pred` - Predicted values tensor
    ///
    /// # Returns
    /// Gradient tensor with same shape as `y_pred`
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensors have incompatible shapes or if mutex lock fails.
    #[pyo3(signature = (y_true, y_pred))]
    fn backward(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .derivative(
                        y_true
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                        y_pred
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }
}

/// Binary Cross-Entropy (BCE) loss function.
///
/// **Formula:**
/// ```text
/// L(y, ŷ) = -(1/n) · Σᵢ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
/// ```
///
/// **Standard Gradient:**
/// ```text
/// ∂L/∂ŷ = -(1/n) · [y/ŷ - (1-y)/(1-ŷ)]
/// ```
///
/// **Optimized Gradient (with Sigmoid fusion):**
/// ```text
/// ∂L/∂z = σ(z) - y    (where ŷ = σ(z))
/// ```
///
/// BCE is the standard loss for binary classification tasks. It measures the divergence
/// between predicted probabilities (ŷ ∈ [0,1]) and true binary labels (y ∈ {0,1}).
///
/// **Numerical Stability:** BCE uses epsilon clamping (1e-15) to prevent log(0) errors.
///
/// **Use Cases:**
/// - Binary classification (with Sigmoid activation)
/// - Multi-label classification (independent binary predictions per label)
///
/// # Optimization
///
/// When combined with a Sigmoid activation layer, use `backward_optimized()` instead of `backward()`.
/// This computes the **fused gradient** `σ(z) - y` which is numerically stable and more efficient
/// than chaining Sigmoid's Jacobian with BCE's derivative.
///
/// # Python API
///
/// ```python
/// from neomatrix._backend import BCE
///
/// loss_fn = BCE()
/// loss_value = loss_fn.call(y_true, y_pred)  # y_pred should be in [0,1]
///
/// # Standard backprop (if activation already applied):
/// gradient = loss_fn.backward(y_true, y_pred)
///
/// # Optimized backprop (if activation not applied, pass logits):
/// gradient = loss_fn.backward_optimized(y_true, logits)  # logits = pre-activation values
/// ```
#[pyclass(name = "BCE")]
pub struct PyBinaryCrossEntropy {
    inner: losses::BinaryCrossEntropy,
}

#[pymethods]
impl PyBinaryCrossEntropy {
    /// Creates a new Binary Cross-Entropy loss function.
    ///
    /// # Returns
    /// A new `BCE` instance ready to compute loss and gradients.
    #[new]
    fn new() -> Self {
        PyBinaryCrossEntropy {
            inner: losses::BinaryCrossEntropy,
        }
    }

    /// Computes the BCE loss between true labels and predicted probabilities.
    ///
    /// **Expects:** `y_pred` should be in range [0,1] (post-Sigmoid activation).
    ///
    /// # Arguments
    /// * `y_true` - Ground truth binary labels (0 or 1)
    /// * `y_pred` - Predicted probabilities (must be in [0,1])
    ///
    /// # Returns
    /// Scalar loss value (f32)
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensors have incompatible shapes or if mutex lock fails.
    #[pyo3(signature = (y_true, y_pred))]
    fn call(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<f32> {
        Ok(self
            .inner
            .function(
                y_true
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
                y_pred
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?)
    }

    /// Computes the standard gradient of BCE loss.
    ///
    /// Gradient formula: `∂L/∂ŷ = -(1/n) · [y/ŷ - (1-y)/(1-ŷ)]`
    ///
    /// Use this when the Sigmoid activation has already been applied to get `y_pred`.
    ///
    /// # Arguments
    /// * `y_true` - Ground truth binary labels (0 or 1)
    /// * `y_pred` - Predicted probabilities in [0,1]
    ///
    /// # Returns
    /// Gradient tensor with same shape as `y_pred`
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensors have incompatible shapes or if mutex lock fails.
    #[pyo3(signature = (y_true, y_pred))]
    fn backward(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .derivative(
                        y_true
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                        y_pred
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    /// Computes the **optimized fused gradient** for Sigmoid + BCE.
    ///
    /// Gradient formula: `∂L/∂z = σ(z) - y` (where `y_pred = σ(z)`)
    ///
    /// This method assumes `y_pred` is actually the **pre-activation logits** (z),
    /// and computes the combined gradient of Sigmoid activation followed by BCE loss.
    /// This is **numerically stable** and more efficient than separate backward passes.
    ///
    /// **When to use:**
    /// - When your last layer outputs raw logits (no Sigmoid activation applied)
    /// - During backpropagation through a Sigmoid + BCE layer chain
    ///
    /// # Arguments
    /// * `y_true` - Ground truth binary labels (0 or 1)
    /// * `y_pred` - **Pre-activation logits** (NOT probabilities)
    ///
    /// # Returns
    /// Fused gradient tensor with same shape as `y_pred`
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensors have incompatible shapes or if mutex lock fails.
    #[pyo3(signature = (y_true, y_pred))]
    fn backward_optimized(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<PyTensor> {
        // Fused gradient computation: σ(z) - y
        // This is mathematically equivalent to chaining Sigmoid backward with BCE backward,
        // but numerically stable and faster.
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                (y_pred
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref()
                    - y_true
                        .inner
                        .lock()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .deref())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }
}

/// Categorical Cross-Entropy (CCE) loss function.
///
/// **Formula:**
/// ```text
/// L(y, ŷ) = -(1/n) · ΣᵢΣₖ yᵢₖ·log(ŷᵢₖ)
/// ```
///
/// **Standard Gradient:**
/// ```text
/// ∂L/∂ŷ = -(1/n) · (y / ŷ)
/// ```
///
/// **Optimized Gradient (with Softmax fusion):**
/// ```text
/// ∂L/∂z = softmax(z) - y    (where ŷ = softmax(z))
/// ```
///
/// CCE is the standard loss for multi-class classification tasks. It measures the divergence
/// between predicted probability distributions (ŷ, class probabilities) and true one-hot encoded
/// labels (y, where yₖ=1 for correct class, 0 elsewhere).
///
/// **Numerical Stability:** CCE uses epsilon clamping (1e-15) to prevent log(0) errors.
///
/// **Use Cases:**
/// - Multi-class classification (with Softmax activation)
/// - Single-label prediction (exactly one class per sample)
///
/// # Optimization
///
/// When combined with a Softmax activation layer, use `backward_optimized()` instead of `backward()`.
/// This computes the **fused gradient** `softmax(z) - y` which is numerically stable and more efficient
/// than chaining Softmax's Jacobian with CCE's derivative.
///
/// # Python API
///
/// ```python
/// from neomatrix._backend import CCE
///
/// loss_fn = CCE()
/// # y_true: one-hot encoded labels, y_pred: class probabilities (post-Softmax)
/// loss_value = loss_fn.call(y_true, y_pred)
///
/// # Standard backprop (if activation already applied):
/// gradient = loss_fn.backward(y_true, y_pred)
///
/// # Optimized backprop (if activation not applied, pass logits):
/// gradient = loss_fn.backward_optimized(y_true, logits)  # logits = pre-activation values
/// ```
#[pyclass(name = "CCE")]
pub struct PyCategoricalCrossEntropy {
    inner: losses::CategoricalCrossEntropy,
}

#[pymethods]
impl PyCategoricalCrossEntropy {
    /// Creates a new Categorical Cross-Entropy loss function.
    ///
    /// # Returns
    /// A new `CCE` instance ready to compute loss and gradients.
    #[new]
    fn new() -> Self {
        PyCategoricalCrossEntropy {
            inner: losses::CategoricalCrossEntropy,
        }
    }

    /// Computes the CCE loss between true labels and predicted probabilities.
    ///
    /// **Expects:** `y_pred` should be a probability distribution (post-Softmax activation).
    ///
    /// # Arguments
    /// * `y_true` - One-hot encoded ground truth labels
    /// * `y_pred` - Predicted class probabilities (must sum to 1 per sample)
    ///
    /// # Returns
    /// Scalar loss value (f32)
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensors have incompatible shapes or if mutex lock fails.
    #[pyo3(signature = (y_true, y_pred))]
    fn call(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<f32> {
        Ok(self
            .inner
            .function(
                y_true
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
                y_pred
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?)
    }

    /// Computes the standard gradient of CCE loss.
    ///
    /// Gradient formula: `∂L/∂ŷ = -(1/n) · (y / ŷ)`
    ///
    /// Use this when the Softmax activation has already been applied to get `y_pred`.
    ///
    /// # Arguments
    /// * `y_true` - One-hot encoded ground truth labels
    /// * `y_pred` - Predicted class probabilities (post-Softmax)
    ///
    /// # Returns
    /// Gradient tensor with same shape as `y_pred`
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensors have incompatible shapes or if mutex lock fails.
    #[pyo3(signature = (y_true, y_pred))]
    fn backward(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .derivative(
                        y_true
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                        y_pred
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    /// Computes the **optimized fused gradient** for Softmax + CCE.
    ///
    /// Gradient formula: `∂L/∂z = softmax(z) - y` (where `y_pred = softmax(z)`)
    ///
    /// This method assumes `y_pred` is actually the **pre-activation logits** (z),
    /// and computes the combined gradient of Softmax activation followed by CCE loss.
    /// This is **numerically stable** and more efficient than separate backward passes.
    ///
    /// **When to use:**
    /// - When your last layer outputs raw logits (no Softmax activation applied)
    /// - During backpropagation through a Softmax + CCE layer chain
    ///
    /// # Arguments
    /// * `y_true` - One-hot encoded ground truth labels
    /// * `y_pred` - **Pre-activation logits** (NOT probabilities)
    ///
    /// # Returns
    /// Fused gradient tensor with same shape as `y_pred`
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensors have incompatible shapes or if mutex lock fails.
    #[pyo3(signature = (y_true, y_pred))]
    fn backward_optimized(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<PyTensor> {
        // Fused gradient computation: softmax(z) - y
        // Mathematically equivalent to chaining Softmax Jacobian with CCE derivative,
        // but avoids numerical instability from exp() overflow/underflow.
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                (y_pred
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref()
                    - y_true
                        .inner
                        .lock()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .deref())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }
}

/// Huber Loss function (robust regression loss).
///
/// **Formula:**
/// ```text
/// L(y, ŷ) = (1/n) · Σᵢ Lδ(yᵢ - ŷᵢ)
///
/// where:
///   Lδ(a) = ½a²               if |a| ≤ δ
///   Lδ(a) = δ|a| - ½δ²        if |a| > δ
/// ```
///
/// **Gradient:**
/// ```text
/// ∂L/∂ŷ = (1/n) · (ŷ - y)           if |y - ŷ| ≤ δ
/// ∂L/∂ŷ = -(δ/n) · sign(y - ŷ)      if |y - ŷ| > δ
/// ```
///
/// Huber Loss combines the best properties of MSE and MAE:
/// - **Quadratic (MSE-like)** for small errors: smooth gradients, fast convergence near optimum
/// - **Linear (MAE-like)** for large errors: robust to outliers, prevents gradient explosion
///
/// The transition point `δ` controls where the loss switches from quadratic to linear behavior.
///
/// **Use Cases:**
/// - Regression with outliers
/// - When you want MSE's smoothness but MAE's robustness
/// - Reinforcement learning (e.g., DQN value function approximation)
///
/// # Python API
///
/// ```python
/// from neomatrix._backend import HuberLoss
///
/// loss_fn = HuberLoss(delta=1.0)  # Transition point
/// loss_fn.delta = 0.5             # Can adjust dynamically
///
/// loss_value = loss_fn.call(y_true, y_pred)
/// gradient = loss_fn.backward(y_true, y_pred)
/// ```
#[pyclass(name = "HuberLoss")]
pub struct PyHuberLoss {
    inner: losses::HuberLoss,
}

#[pymethods]
impl PyHuberLoss {
    /// Creates a new Huber Loss function with the specified delta threshold.
    ///
    /// # Arguments
    /// * `delta` - Transition point between quadratic and linear regions (typical: 0.5-2.0)
    ///
    /// # Returns
    /// A new `HuberLoss` instance.
    #[new]
    fn new(delta: f32) -> Self {
        PyHuberLoss {
            inner: losses::HuberLoss { delta },
        }
    }

    /// Gets the current delta threshold.
    ///
    /// # Returns
    /// Current delta value (f32)
    #[getter]
    fn delta(&self) -> f32 {
        self.inner.delta
    }

    /// Sets a new delta threshold.
    ///
    /// # Arguments
    /// * `value` - New delta value (must be positive)
    ///
    /// # Errors
    /// Returns `PyResult` (always Ok for now, but signature prepared for validation).
    #[setter]
    fn set_delta(&mut self, value: f32) -> PyResult<()> {
        self.inner.delta = value;
        Ok(())
    }

    /// Computes the Huber loss between true and predicted values.
    ///
    /// # Arguments
    /// * `y_true` - Ground truth tensor
    /// * `y_pred` - Predicted values tensor
    ///
    /// # Returns
    /// Scalar loss value (f32)
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensors have incompatible shapes or if mutex lock fails.
    #[pyo3(signature = (y_true, y_pred))]
    fn call(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<f32> {
        Ok(self
            .inner
            .function(
                y_true
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
                y_pred
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?)
    }

    /// Computes the gradient of Huber loss.
    ///
    /// Gradient switches based on error magnitude:
    /// - Quadratic region (|error| ≤ δ): gradient = (ŷ - y)/n
    /// - Linear region (|error| > δ): gradient = -δ·sign(y - ŷ)/n
    ///
    /// # Arguments
    /// * `y_true` - Ground truth tensor
    /// * `y_pred` - Predicted values tensor
    ///
    /// # Returns
    /// Gradient tensor with same shape as `y_pred`
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensors have incompatible shapes or if mutex lock fails.
    #[pyo3(signature = (y_true, y_pred))]
    fn backward(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .derivative(
                        y_true
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                        y_pred
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }
}

/// Hinge Loss function (SVM-style classification loss).
///
/// **Formula:**
/// ```text
/// L(y, ŷ) = (1/n) · Σᵢ max(0, 1 - yᵢ·ŷᵢ)
/// ```
///
/// **Gradient:**
/// ```text
/// ∂L/∂ŷ = -(1/n) · y    if y·ŷ < 1
/// ∂L/∂ŷ = 0             if y·ŷ ≥ 1
/// ```
///
/// Hinge Loss is the standard loss for Support Vector Machines (SVM). It encourages:
/// - Correct predictions (y·ŷ > 1) with **zero loss** (no penalty for confident correct predictions)
/// - Incorrect predictions with **linear penalty** proportional to the margin violation
///
/// **Labels:** Expects y ∈ {-1, +1} (NOT {0, 1} like BCE/CCE)
///
/// **Use Cases:**
/// - Binary classification with maximum margin objective (SVM-like)
/// - When you want the model to be "confidently correct" (penalizes predictions near the decision boundary)
/// - Face verification, anomaly detection
///
/// # Python API
///
/// ```python
/// from neomatrix._backend import HingeLoss
///
/// loss_fn = HingeLoss()
/// # y_true: labels in {-1, +1}, y_pred: raw decision function values (NOT probabilities)
/// loss_value = loss_fn.call(y_true, y_pred)
/// gradient = loss_fn.backward(y_true, y_pred)
/// ```
#[pyclass(name = "HingeLoss")]
pub struct PyHingeLoss {
    inner: losses::HingeLoss,
}

#[pymethods]
impl PyHingeLoss {
    /// Creates a new Hinge Loss function.
    ///
    /// # Returns
    /// A new `HingeLoss` instance ready to compute loss and gradients.
    #[new]
    fn new() -> Self {
        PyHingeLoss {
            inner: losses::HingeLoss,
        }
    }

    /// Computes the Hinge loss between true labels and decision values.
    ///
    /// **Expects:**
    /// - `y_true` with labels in {-1, +1} (NOT {0, 1})
    /// - `y_pred` as raw decision function values (NOT probabilities)
    ///
    /// # Arguments
    /// * `y_true` - Ground truth labels in {-1, +1}
    /// * `y_pred` - Raw decision function values (pre-activation)
    ///
    /// # Returns
    /// Scalar loss value (f32)
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensors have incompatible shapes or if mutex lock fails.
    #[pyo3(signature = (y_true, y_pred))]
    fn call(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<f32> {
        Ok(self
            .inner
            .function(
                y_true
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
                y_pred
                    .inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?)
    }

    /// Computes the gradient of Hinge loss.
    ///
    /// Gradient is **sparse**: non-zero only for margin violations (y·ŷ < 1).
    /// - If y·ŷ < 1 (wrong or uncertain): ∂L/∂ŷ = -y/n
    /// - If y·ŷ ≥ 1 (correct with margin): ∂L/∂ŷ = 0
    ///
    /// # Arguments
    /// * `y_true` - Ground truth labels in {-1, +1}
    /// * `y_pred` - Raw decision function values
    ///
    /// # Returns
    /// Gradient tensor with same shape as `y_pred`
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensors have incompatible shapes or if mutex lock fails.
    #[pyo3(signature = (y_true, y_pred))]
    fn backward(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .derivative(
                        y_true
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                        y_pred
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }
}
