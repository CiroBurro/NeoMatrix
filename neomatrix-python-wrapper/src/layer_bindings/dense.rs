//! Python bindings for the Dense (fully-connected) layer.
//!
//! Provides a trainable layer with weight matrix, bias vector, and backpropagation support.
//! See `neomatrix_core::layers::dense` for mathematical formulas and implementation details.

use std::{
    ops::Deref,
    sync::{Arc, Mutex},
};

use neomatrix_core::layers::{dense::Dense, Layer};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    layer_bindings::init::PyInit, optimizer_bindings::PyParametersRef, tensor_bindings::PyTensor,
};

/// Python wrapper for Dense layer (fully-connected neural network layer).
///
/// Performs linear transformation: Y = X·W + b
/// where W is the weight matrix and b is the bias vector.
///
/// # Parameters
///
/// During training, the layer maintains:
/// - Weights (W): shape (input_size, output_size)
/// - Biases (b): shape (output_size,)
/// - Weight gradients (∇W): accumulated during backpropagation
/// - Bias gradients (∇b): accumulated during backpropagation
///
/// All parameters are shared via Arc<Mutex<Tensor>> with registered optimizers.
///
/// # Usage
///
/// ```python
/// # Create layer with Xavier initialization
/// layer = Dense(input_size=784, output_size=128, init=Init.Xavier)
///
/// # Forward pass
/// output = layer.forward(input, training=True)
///
/// # Backward pass (accumulates gradients)
/// grad_input = layer.backward(grad_output)
///
/// # Get parameters for optimizer registration
/// params = layer.get_parameters()
/// optimizer.register_params([params])
/// ```
#[pyclass(name = "Dense")]
#[derive(Clone, Debug)]
pub struct PyDense {
    pub inner: Dense,
}

#[pymethods]
impl PyDense {
    /// Create a new Dense layer.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features (fan-in)
    /// * `output_size` - Number of output features (fan-out)
    /// * `init` - Weight initialization strategy (Init.He, Init.Xavier, Init.Random). Defaults to Xavier.
    /// * `range_start` - Start of uniform range (required only for Init.Random)
    /// * `range_end` - End of uniform range (required only for Init.Random)
    ///
    /// # Returns
    ///
    /// A new Dense layer with initialized weights and zero biases.
    ///
    /// # Example
    ///
    /// ```python
    /// # Using He initialization for ReLU network
    /// layer = Dense(input_size=784, output_size=128, init=Init.He)
    ///
    /// # Using Random initialization with custom range
    /// layer = Dense(input_size=100, output_size=50, init=Init.Random, range_start=-0.5, range_end=0.5)
    /// ```
    #[new]
    #[pyo3(signature = (input_size, output_size, init=None, range_start=None, range_end=None))]
    pub fn new(
        input_size: usize,
        output_size: usize,
        init: Option<PyInit>,
        range_start: Option<f32>,
        range_end: Option<f32>,
    ) -> Self {
        // Convert Option<f32> pair to Option<Range<f32>>
        let rg = match (range_start, range_end) {
            (Some(start), Some(end)) => Some(start..end),
            _ => None,
        };
        PyDense {
            inner: Dense::new(input_size, output_size, init.map(|i| i.inner), rg),
        }
    }

    /// Forward pass: compute Y = X·W + b.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (..., input_size)
    /// * `training` - If True, caches input for backward pass
    ///
    /// # Returns
    ///
    /// Output tensor of shape (..., output_size)
    #[pyo3(signature = (input, training))]
    pub fn forward(&mut self, input: PyTensor, training: bool) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .forward(
                        &input
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

    /// Backward pass: compute input gradients and accumulate parameter gradients.
    ///
    /// Computes three gradients:
    /// - ∇W = X^T · ∇Y (accumulated into self.w_grads)
    /// - ∇b = sum(∇Y) (accumulated into self.b_grads)
    /// - ∇X = ∇Y · W^T (returned)
    ///
    /// **Note:** Gradients accumulate across calls. Optimizer must call `zero_grad()` before each training step.
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
                        &output_gradient
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    /// Get shared references to layer parameters for optimizer registration.
    ///
    /// Returns Arc<Mutex<Tensor>> references to weights, biases, and their gradients.
    /// This allows optimizers to update parameters in-place without re-fetching from layers.
    ///
    /// Gradients are initialized to zero at layer construction, so this method can be called
    /// immediately after `__new__()` without requiring a forward/backward pass first.
    ///
    /// # Returns
    ///
    /// ParametersRef containing shared references to all layer parameters:
    /// - `weights`: Weight matrix [input_size, output_size]
    /// - `biases`: Bias vector [output_size]
    /// - `w_grads`: Weight gradients [input_size, output_size] (initially zeros)
    /// - `b_grads`: Bias gradients [output_size] (initially zeros)
    ///
    /// # Example
    ///
    /// ```python
    /// # Create layer and immediately get parameters (no forward/backward needed)
    /// layer = Dense(input_size=784, output_size=128)
    /// params = layer.get_parameters()
    /// optimizer.register_params([params])
    ///
    /// # Training loop
    /// optimizer.zero_grad()
    /// output = layer.forward(input, training=True)
    /// # ... compute loss ...
    /// grad = layer.backward(grad_output)
    /// optimizer.step()  # Updates weights using accumulated gradients
    /// ```
    pub fn get_parameters(&mut self) -> PyResult<PyParametersRef> {
        let inner = self
            .inner
            .get_parameters()
            .ok_or(PyRuntimeError::new_err("No gradients computed yet"))?;

        Ok(PyParametersRef { inner })
    }
}
