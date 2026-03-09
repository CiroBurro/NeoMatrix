use neomatrix_core::layers::{dense::Dense, Layer};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{layer_bindings::init::PyInit, tensor_bindings::PyTensor};

#[pyclass(name = "Dense")]
#[derive(Clone, Debug)]
pub struct PyDense {
    pub inner: Dense,
}

#[pymethods]
impl PyDense {
    #[new]
    #[pyo3(signature = (input_size, output_size, init=None, range_start=None, range_end=None))]
    pub fn new(
        input_size: usize,
        output_size: usize,
        init: Option<PyInit>,
        range_start: Option<f32>,
        range_end: Option<f32>,
    ) -> Self {
        let rg = match (range_start, range_end) {
            (Some(start), Some(end)) => Some(start..end),
            _ => None,
        };
        PyDense {
            inner: Dense::new(input_size, output_size, init.map(|i| i.inner), rg),
        }
    }

    pub fn forward(&mut self, input: PyTensor, training: bool) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self
                .inner
                .forward(&input.inner, training)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    pub fn backward(&mut self, output_gradient: PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self
                .inner
                .backward(&output_gradient.inner)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    /// Returns cloned (param, gradient) pairs.
    /// Returns an empty list if no gradients have been computed yet (before backward pass).
    pub fn get_params_and_grads(&mut self) -> Vec<(PyTensor, PyTensor)> {
        self.inner
            .get_params_and_grads()
            .unwrap_or_default()
            .into_iter()
            .map(|(param, grad)| {
                (
                    PyTensor {
                        inner: param.clone(),
                    },
                    PyTensor {
                        inner: grad.clone(),
                    },
                )
            })
            .collect()
    }
}
