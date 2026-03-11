use crate::optimizer_bindings::PyParametersRef;
use neomatrix_core::optimizers::{gradient_descent::GradientDescent, Optimizer};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pyclass(name = "GradientDescent")]
pub struct PyGD {
    pub inner: GradientDescent,
}

#[pymethods]
impl PyGD {
    #[new]
    pub fn new(learning_rate: f32) -> Self {
        Self {
            inner: GradientDescent::new(learning_rate, Vec::new()),
        }
    }

    pub fn register_params(&mut self, params: Vec<Bound<'_, PyParametersRef>>) {
        self.inner
            .register_params(params.iter().map(|p| p.borrow().inner.clone()).collect());
    }

    pub fn step(&mut self) -> PyResult<()> {
        self.inner
            .step()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn zero_grad(&mut self) -> PyResult<()> {
        self.inner
            .zero_grad()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}
