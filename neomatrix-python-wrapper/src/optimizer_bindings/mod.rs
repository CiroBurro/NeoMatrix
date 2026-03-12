//! Python bindings for optimizer implementations.
//!
//! This module provides Python wrappers for stateful optimizers that update neural network
//! parameters. Optimizers follow the PyTorch-style API with parameter registration, step(),
//! and zero_grad() methods.

use neomatrix_core::optimizers::ParametersRef;
use pyo3::prelude::*;

use crate::tensor_bindings::PyTensor;

pub mod gradient_descent;

/// Python wrapper for ParametersRef - shared parameter container.
///
/// This struct holds Arc<Mutex<Tensor>> references to a layer's weights, biases,
/// and their gradients. It enables optimizers and layers to share ownership of the same
/// tensors, allowing optimizers to update parameters in-place after backpropagation.
///
/// # Thread Safety
///
/// All tensors are wrapped in Arc<Mutex<>> for safe shared ownership. Lock contention is
/// negligible in practice since:
/// - Layers lock during forward/backward passes
/// - Optimizers lock during step()/zero_grad()
/// - These operations never overlap (sequential by design)
///
/// # Typical Usage
///
/// Users don't construct this directly — it's returned by `Dense.get_parameters()`:
///
/// ```python
/// layer = Dense(input_size=784, output_size=10)
/// params = layer.get_parameters()  # Returns ParametersRef
/// optimizer.register_params([params])
/// ```
#[pyclass(name = "ParametersRef")]
pub struct PyParametersRef {
    pub inner: ParametersRef,
}

#[pymethods]
impl PyParametersRef {
    /// Create a new ParametersRef from four tensors.
    ///
    /// **NOTE:** This constructor is rarely used directly. Layers automatically create
    /// ParametersRef instances and return them via `get_parameters()`.
    ///
    /// # Arguments
    ///
    /// * `weights` - Layer weight matrix (Arc<Mutex<Tensor>>)
    /// * `biases` - Layer bias vector (Arc<Mutex<Tensor>>)
    /// * `w_grads` - Weight gradients (Arc<Mutex<Tensor>>)
    /// * `b_grads` - Bias gradients (Arc<Mutex<Tensor>>)
    ///
    /// # Returns
    ///
    /// A new ParametersRef wrapping the provided tensors.
    #[new]
    #[pyo3(signature = (weights, biases, w_grads, b_grads))]
    pub fn new(weights: PyTensor, biases: PyTensor, w_grads: PyTensor, b_grads: PyTensor) -> Self {
        Self {
            inner: ParametersRef {
                weights: weights.inner,
                biases: biases.inner,
                w_grads: w_grads.inner,
                b_grads: b_grads.inner,
            },
        }
    }
}
