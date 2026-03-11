use std::{
    ops::Deref,
    sync::{Arc, Mutex},
};

use crate::tensor_bindings::PyTensor;
use neomatrix_core::layers::{activations, Layer};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pyclass(name = "ReLU")]
pub struct PyReLU {
    pub inner: activations::ReLu,
}
#[pymethods]
impl PyReLU {
    #[new]
    pub fn new() -> Self {
        PyReLU {
            inner: activations::ReLu::new(),
        }
    }

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

    fn __repr__(&self) -> String {
        format!("ReLU")
    }
}

#[pyclass(name = "Sigmoid")]
pub struct PySigmoid {
    pub inner: activations::Sigmoid,
}
#[pymethods]
impl PySigmoid {
    #[new]
    pub fn new() -> Self {
        PySigmoid {
            inner: activations::Sigmoid::new(),
        }
    }

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

    pub fn backward_optimized(&self, output_gradient: PyTensor) -> PyResult<PyTensor> {
        Ok(output_gradient)
    }

    fn __repr__(&self) -> String {
        format!("Sigmoid")
    }
}

#[pyclass(name = "Tanh")]
pub struct PyTanh {
    pub inner: activations::Tanh,
}
#[pymethods]
impl PyTanh {
    #[new]
    pub fn new() -> Self {
        PyTanh {
            inner: activations::Tanh::new(),
        }
    }

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

    fn __repr__(&self) -> String {
        format!("Tanh")
    }
}

#[pyclass(name = "Softmax")]
pub struct PySoftmax {
    pub inner: activations::Softmax,
}
#[pymethods]
impl PySoftmax {
    #[new]
    pub fn new() -> Self {
        PySoftmax {
            inner: activations::Softmax::new(),
        }
    }

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

    pub fn backward_optimized(&self, output_gradient: PyTensor) -> PyResult<PyTensor> {
        Ok(output_gradient)
    }

    fn __repr__(&self) -> String {
        format!("Softmax")
    }
}
