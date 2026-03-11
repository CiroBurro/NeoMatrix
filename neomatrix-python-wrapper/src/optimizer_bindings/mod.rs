use neomatrix_core::optimizers::ParametersRef;
use pyo3::prelude::*;

use crate::tensor_bindings::PyTensor;

pub mod gradient_descent;

#[pyclass(name = "ParametersRef")]
pub struct PyParametersRef {
    pub inner: ParametersRef,
}

#[pymethods]
impl PyParametersRef {
    #[new]
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
