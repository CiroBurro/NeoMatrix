//! Python bindings for activation layer wrappers.
//!
//! Provides stateful activation layers (ReLU, Sigmoid, Tanh, Softmax) that cache
//! inputs/outputs during forward passes for efficient backpropagation. All layers
//! support both classical gradient computation and optimized fused backprop when
//! combined with compatible loss functions (BCE, CCE).

use std::{
    ops::Deref,
    sync::{Arc, Mutex},
};

use crate::tensor_bindings::PyTensor;
use neomatrix_core::layers::{activations, Layer};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

/// Python wrapper for ReLU activation layer.
///
/// Rectified Linear Unit: f(x) = max(0, x)
/// Derivative: f'(x) = 1 if x > 0, else 0
///
/// Caches input during forward pass for efficient backpropagation.
///
/// # Usage
///
/// ```python
/// relu = ReLU()
/// y = relu.forward(x, training=True)
/// grad_x = relu.backward(grad_y)
/// ```
#[pyclass(name = "ReLU")]
pub struct PyReLU {
    pub inner: activations::ReLu,
}
#[pymethods]
impl PyReLU {
    /// Create a new ReLU activation layer.
    #[new]
    pub fn new() -> Self {
        PyReLU {
            inner: activations::ReLu::new(),
        }
    }

    /// Forward pass: apply ReLU activation element-wise.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (any shape)
    /// * `training` - If True, caches input for backward pass
    ///
    /// # Returns
    ///
    /// Activated tensor with same shape as input: max(0, input)
    #[pyo3(signature = (input, training))]
    pub fn forward(&mut self, input: PyTensor, training: bool) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .forward(
                        input
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                        training,
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    /// Backward pass: compute input gradients using cached values.
    ///
    /// Uses classical ReLU derivative: passes gradient through where input > 0, zeros elsewhere.
    ///
    /// # Arguments
    ///
    /// * `output_gradient` - Gradient from next layer (∂Loss/∂output)
    ///
    /// # Returns
    ///
    /// Input gradient (∂Loss/∂input) with same shape as cached input
    #[pyo3(signature = (output_gradient))]
    pub fn backward(&mut self, output_gradient: PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .backward(
                        output_gradient
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    /// Python repr: 'ReLU'
    fn __repr__(&self) -> String {
        format!("ReLU")
    }
}

/// Python wrapper for Sigmoid activation layer.
///
/// Sigmoid: f(x) = 1 / (1 + exp(-x))
/// Derivative: f'(x) = σ(x) · (1 - σ(x))
///
/// Caches output during forward pass for efficient backpropagation.
/// Supports optimized gradient passthrough for Binary Cross-Entropy loss via `backward_optimized()`.
///
/// # Usage
///
/// ```python
/// # Standard usage
/// sigmoid = Sigmoid()
/// y = sigmoid.forward(x, training=True)
/// grad_x = sigmoid.backward(grad_y)
///
/// # Fused optimization with BCE loss
/// grad_x = sigmoid.backward_optimized(grad_from_bce)  # Just passes through
/// ```
#[pyclass(name = "Sigmoid")]
pub struct PySigmoid {
    pub inner: activations::Sigmoid,
}
#[pymethods]
impl PySigmoid {
    /// Create a new Sigmoid activation layer.
    #[new]
    pub fn new() -> Self {
        PySigmoid {
            inner: activations::Sigmoid::new(),
        }
    }

    /// Forward pass: apply Sigmoid activation element-wise.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (any shape)
    /// * `training` - If True, caches output for backward pass
    ///
    /// # Returns
    ///
    /// Activated tensor: 1 / (1 + exp(-input))
    #[pyo3(signature = (input, training))]
    pub fn forward(&mut self, input: PyTensor, training: bool) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .forward(
                        input
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                        training,
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    /// Backward pass: classical Sigmoid derivative.
    ///
    /// Computes gradient using cached output: ∂Loss/∂input = ∂Loss/∂output · σ(1-σ)
    ///
    /// # Arguments
    ///
    /// * `output_gradient` - Gradient from next layer (∂Loss/∂output)
    ///
    /// # Returns
    ///
    /// Input gradient (∂Loss/∂input)
    #[pyo3(signature = (output_gradient))]
    pub fn backward(&mut self, output_gradient: PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .backward(
                        output_gradient
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    /// Optimized backward for fused Sigmoid + BCE loss.
    ///
    /// When using Binary Cross-Entropy loss with `backward_optimized()`, the loss function
    /// returns the pre-computed combined gradient: σ(z) - y_true. This method just passes
    /// it through without modification, avoiding redundant computation of the Jacobian.
    ///
    /// **Usage:** Only call this when loss function used `backward_optimized()`.
    ///
    /// # Arguments
    ///
    /// * `output_gradient` - Pre-computed combined gradient from BCE loss
    ///
    /// # Returns
    ///
    /// Same gradient (passthrough)
    #[pyo3(signature = (output_gradient))]
    pub fn backward_optimized(&self, output_gradient: PyTensor) -> PyResult<PyTensor> {
        Ok(output_gradient)
    }

    /// Python repr: 'Sigmoid'
    fn __repr__(&self) -> String {
        format!("Sigmoid")
    }
}

/// Python wrapper for Tanh activation layer.
///
/// Hyperbolic Tangent: f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
/// Derivative: f'(x) = 1 - tanh²(x)
///
/// Caches output during forward pass for efficient backpropagation.
///
/// # Usage
///
/// ```python
/// tanh = Tanh()
/// y = tanh.forward(x, training=True)
/// grad_x = tanh.backward(grad_y)
/// ```
#[pyclass(name = "Tanh")]
pub struct PyTanh {
    pub inner: activations::Tanh,
}
#[pymethods]
impl PyTanh {
    /// Create a new Tanh activation layer.
    #[new]
    pub fn new() -> Self {
        PyTanh {
            inner: activations::Tanh::new(),
        }
    }

    /// Forward pass: apply Tanh activation element-wise.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (any shape)
    /// * `training` - If True, caches output for backward pass
    ///
    /// # Returns
    ///
    /// Activated tensor: tanh(input), values in range (-1, 1)
    #[pyo3(signature = (input, training))]
    pub fn forward(&mut self, input: PyTensor, training: bool) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .forward(
                        input
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                        training,
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    /// Backward pass: classical Tanh derivative.
    ///
    /// Computes gradient using cached output: ∂Loss/∂input = ∂Loss/∂output · (1 - tanh²)
    ///
    /// # Arguments
    ///
    /// * `output_gradient` - Gradient from next layer (∂Loss/∂output)
    ///
    /// # Returns
    ///
    /// Input gradient (∂Loss/∂input)
    #[pyo3(signature = (output_gradient))]
    pub fn backward(&mut self, output_gradient: PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .backward(
                        output_gradient
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    /// Python repr: 'Tanh'
    fn __repr__(&self) -> String {
        format!("Tanh")
    }
}

/// Python wrapper for Softmax activation layer.
///
/// Softmax: f(x_i) = exp(x_i) / Σ exp(x_j)
/// Converts logits to probability distribution (sum = 1, all values in (0,1)).
///
/// Uses log-sum-exp trick for numerical stability: softmax(x) = softmax(x - max(x))
///
/// Caches output during forward pass for efficient backpropagation.
/// Supports optimized gradient passthrough for Categorical Cross-Entropy loss via `backward_optimized()`.
///
/// # Usage
///
/// ```python
/// # Standard usage
/// softmax = Softmax()
/// probs = softmax.forward(logits, training=True)
/// grad_logits = softmax.backward(grad_probs)
///
/// # Fused optimization with CCE loss
/// grad_logits = softmax.backward_optimized(grad_from_cce)  # Just passes through
/// ```
#[pyclass(name = "Softmax")]
pub struct PySoftmax {
    pub inner: activations::Softmax,
}
#[pymethods]
impl PySoftmax {
    /// Create a new Softmax activation layer.
    #[new]
    pub fn new() -> Self {
        PySoftmax {
            inner: activations::Softmax::new(),
        }
    }

    /// Forward pass: apply Softmax activation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input logits (any shape, typically 1D or 2D)
    /// * `training` - If True, caches output for backward pass
    ///
    /// # Returns
    ///
    /// Probability distribution: exp(input) / Σ exp(input), sum = 1
    #[pyo3(signature = (input, training))]
    pub fn forward(&mut self, input: PyTensor, training: bool) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .forward(
                        input
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                        training,
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    /// Backward pass: classical Softmax Jacobian.
    ///
    /// Computes full Jacobian matrix: J[i,j] = σ_i(δ_ij - σ_j) where σ = softmax output.
    /// This is expensive but general-purpose (works with any loss function).
    ///
    /// # Arguments
    ///
    /// * `output_gradient` - Gradient from next layer (∂Loss/∂output)
    ///
    /// # Returns
    ///
    /// Input gradient (∂Loss/∂input)
    #[pyo3(signature = (output_gradient))]
    pub fn backward(&mut self, output_gradient: PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .backward(
                        output_gradient
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    /// Optimized backward for fused Softmax + CCE loss.
    ///
    /// When using Categorical Cross-Entropy loss with `backward_optimized()`, the loss function
    /// returns the pre-computed combined gradient: softmax(z) - y_true. This method just passes
    /// it through without modification, avoiding expensive Jacobian computation.
    ///
    /// **Usage:** Only call this when loss function used `backward_optimized()`.
    ///
    /// # Arguments
    ///
    /// * `output_gradient` - Pre-computed combined gradient from CCE loss
    ///
    /// # Returns
    ///
    /// Same gradient (passthrough)
    #[pyo3(signature = (output_gradient))]
    pub fn backward_optimized(&self, output_gradient: PyTensor) -> PyResult<PyTensor> {
        Ok(output_gradient)
    }

    /// Python repr: 'Softmax'
    fn __repr__(&self) -> String {
        format!("Softmax")
    }
}
