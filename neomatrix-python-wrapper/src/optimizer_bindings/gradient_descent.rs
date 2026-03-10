use neomatrix_core::optimizers::{Optimizer, gradient_descent::GradientDescent};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::tensor_bindings::PyTensor;

#[pyclass(name = "GradientDescent")]
pub struct PyGD {
    inner: GradientDescent,
}

#[pymethods]
impl PyGD {
    #[new]
    pub fn new(learning_rate: f32) -> Self {
        Self {
            inner: GradientDescent::new(learning_rate),
        }
    }

    fn update(
        &mut self,
        weights: &mut PyTensor,
        biases: &mut PyTensor,
        w_grads: &PyTensor,
        b_grads: &PyTensor,
        _step: usize,
    ) -> PyResult<()> {
        self.inner
            .update(
                &mut weights.inner,
                &mut biases.inner,
                &w_grads.inner,
                &b_grads.inner,
                _step,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}
