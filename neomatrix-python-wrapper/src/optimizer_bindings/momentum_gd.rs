use crate::optimizer_bindings::PyParametersRef;
use neomatrix_core::optimizers::{momentum_gd::MomentumGD, Optimizer};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
/// Python wrapper for the MomentumGD optimizer.
#[pyclass(name = "MomentumGD")]
pub struct PyMomentumGD {
    inner: MomentumGD,
}

#[pymethods]
impl PyMomentumGD {
    #[new]
    #[pyo3(signature = (learning_rate, momentum))]
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            inner: MomentumGD::new(learning_rate, momentum),
        }
    }

    pub fn register_params(&mut self, params: Vec<Bound<'_, PyParametersRef>>) {
        self.inner.register_params(
            params
                .into_iter()
                .map(|p| p.borrow().inner.clone())
                .collect(),
        );
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
