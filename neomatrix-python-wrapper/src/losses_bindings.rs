use crate::tensor_bindings::PyTensor;
use neomatrix_core::math::losses::{self, LossFunction};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::ops::Deref;
use std::sync::{Arc, Mutex};

#[pyclass(name = "MSE")]
pub struct PyMeanSquaredError {
    inner: losses::MeanSquaredError,
}
#[pymethods]
impl PyMeanSquaredError {
    #[new]
    fn new() -> Self {
        PyMeanSquaredError {
            inner: losses::MeanSquaredError,
        }
    }

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

#[pyclass(name = "MAE")]
pub struct PyMeanAbsoluteError {
    inner: losses::MeanAbsoluteError,
}
#[pymethods]
impl PyMeanAbsoluteError {
    #[new]
    fn new() -> Self {
        PyMeanAbsoluteError {
            inner: losses::MeanAbsoluteError,
        }
    }

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

#[pyclass(name = "BCE")]
pub struct PyBinaryCrossEntropy {
    inner: losses::BinaryCrossEntropy,
}
#[pymethods]
impl PyBinaryCrossEntropy {
    #[new]
    fn new() -> Self {
        PyBinaryCrossEntropy {
            inner: losses::BinaryCrossEntropy,
        }
    }

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

    #[pyo3(signature = (y_true, y_pred))]
    fn backward_optimized(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<PyTensor> {
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

#[pyclass(name = "CCE")]
pub struct PyCategoricalCrossEntropy {
    inner: losses::CategoricalCrossEntropy,
}
#[pymethods]
impl PyCategoricalCrossEntropy {
    #[new]
    fn new() -> Self {
        PyCategoricalCrossEntropy {
            inner: losses::CategoricalCrossEntropy,
        }
    }

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

    #[pyo3(signature = (y_true, y_pred))]
    fn backward_optimized(&self, y_true: &PyTensor, y_pred: &PyTensor) -> PyResult<PyTensor> {
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

#[pyclass(name = "HuberLoss")]
pub struct PyHuberLoss {
    inner: losses::HuberLoss,
}
#[pymethods]
impl PyHuberLoss {
    #[new]
    fn new(delta: f32) -> Self {
        PyHuberLoss {
            inner: losses::HuberLoss { delta },
        }
    }

    #[getter]
    fn delta(&self) -> f32 {
        self.inner.delta
    }

    #[setter]
    fn set_delta(&mut self, value: f32) -> PyResult<()> {
        self.inner.delta = value;
        Ok(())
    }

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

#[pyclass(name = "HingeLoss")]
pub struct PyHingeLoss {
    inner: losses::HingeLoss,
}
#[pymethods]
impl PyHingeLoss {
    #[new]
    fn new() -> Self {
        PyHingeLoss {
            inner: losses::HingeLoss,
        }
    }

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
