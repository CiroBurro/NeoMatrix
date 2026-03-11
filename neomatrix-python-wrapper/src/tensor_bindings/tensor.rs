use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use neomatrix_core::tensor::Tensor;
use numpy::{prelude::*, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::Bound;

use crate::tensor_bindings::tensor_iter::TensorIter;

#[pyclass(name = "Tensor")]
#[derive(Clone, Debug)]
pub struct PyTensor {
    pub inner: Arc<Mutex<Tensor>>,
}

#[derive(FromPyObject)]
enum TensorOrScalar<'py> {
    Tensor(PyRef<'py, PyTensor>),
    Scalar(f32),
}

#[pymethods]
impl PyTensor {
    #[pyo3(signature = (shape, content))]
    #[new]
    pub fn new(shape: Vec<usize>, content: Vec<f32>) -> PyResult<PyTensor> {
        let inner =
            Tensor::new(shape, content).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (shape))]
    pub fn zeros(shape: Vec<usize>) -> PyResult<Self> {
        let inner = Tensor::zeros(shape);
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, start=-1.0, end=1.0))]
    pub fn random(shape: Vec<usize>, start: Option<f32>, end: Option<f32>) -> PyResult<Self> {
        let start = start.unwrap_or(-1.0);
        let end = end.unwrap_or(1.0);
        let inner = Tensor::random(shape, start..end);
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (arr))]
    pub fn from_numpy<'py>(arr: PyReadonlyArrayDyn<'py, f32>) -> PyResult<PyTensor> {
        let owned = arr.as_array().to_owned();
        let inner = Tensor {
            dimension: owned.ndim(),
            shape: owned.shape().to_vec(),
            data: owned,
        };
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Getter method for the data field
    /// It converts the ndarray dynamic array into a numpy PyArrayDyn object for python
    #[getter]
    fn get_data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .data
            .to_pyarray(py))
    }

    /// Setter method for the data field
    /// It allows to set the data field from python with a numpy array
    #[setter]
    fn set_data<'py>(&mut self, arr: PyReadonlyArrayDyn<'py, f32>) -> PyResult<()> {
        let owned = arr.as_array().to_owned();
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .dimension = owned.ndim();
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .shape = owned.shape().to_vec();
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .data = owned;
        Ok(())
    }

    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .data
            .to_pyarray(py))
    }

    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .shape
            .clone())
    }

    #[getter]
    fn ndim(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .dimension)
    }

    #[pyo3(signature = (t))]
    pub fn dot(&self, t: &PyTensor) -> PyResult<PyTensor> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .dot(
                &t.inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    pub fn transpose(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref()
                    .transpose()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    pub fn transpose_inplace(&mut self) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .deref_mut()
            .transpose_inplace()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (shape))]
    pub fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref()
                    .reshape(shape)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    pub fn reshape_inplace(&mut self, shape: Vec<usize>) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .deref_mut()
            .reshape_inplace(shape)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn flatten(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref()
                    .flatten(),
            )),
        })
    }

    pub fn flatten_inplace(&mut self) -> PyResult<()> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .deref_mut()
            .flatten_inplace())
    }

    #[pyo3(signature = (t, axis))]
    pub fn push(&mut self, t: &PyTensor, axis: usize) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .deref_mut()
            .push(
                &t.inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
                axis,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[staticmethod]
    #[pyo3(signature = (tensors, axis))]
    pub fn cat(tensors: Vec<PyRef<'_, PyTensor>>, axis: usize) -> PyResult<PyTensor> {
        let inner_tensors: Vec<Tensor> = tensors
            .iter()
            .map(|t| -> PyResult<Tensor> {
                Ok(t.inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref()
                    .clone())
            })
            .collect::<PyResult<Vec<Tensor>>>()?;

        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                Tensor::cat(inner_tensors, axis)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )),
        })
    }

    #[pyo3(signature = (t))]
    pub fn push_row(&mut self, t: &PyTensor) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .deref_mut()
            .push_row(
                &t.inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (t))]
    pub fn push_column(&mut self, t: &PyTensor) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .deref_mut()
            .push_column(
                &t.inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __add__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        match other {
            TensorOrScalar::Tensor(t) => Ok(PyTensor {
                inner: Arc::new(Mutex::new(
                    (self
                        .inner
                        .lock()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .deref()
                        + t.inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                )),
            }),
            TensorOrScalar::Scalar(scalar) => Ok(PyTensor {
                inner: Arc::new(Mutex::new(
                    self.inner
                        .lock()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .deref()
                        + scalar,
                )),
            }),
        }
    }

    fn __sub__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        match other {
            TensorOrScalar::Tensor(t) => Ok(PyTensor {
                inner: Arc::new(Mutex::new(
                    (self
                        .inner
                        .lock()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .deref()
                        - t.inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                )),
            }),
            TensorOrScalar::Scalar(scalar) => Ok(PyTensor {
                inner: Arc::new(Mutex::new(
                    self.inner
                        .lock()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .deref()
                        - scalar,
                )),
            }),
        }
    }

    fn __mul__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        match other {
            TensorOrScalar::Tensor(t) => Ok(PyTensor {
                inner: Arc::new(Mutex::new(
                    (self
                        .inner
                        .lock()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .deref()
                        * t.inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                )),
            }),
            TensorOrScalar::Scalar(scalar) => Ok(PyTensor {
                inner: Arc::new(Mutex::new(
                    self.inner
                        .lock()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .deref()
                        * scalar,
                )),
            }),
        }
    }

    fn __truediv__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        match other {
            TensorOrScalar::Tensor(t) => Ok(PyTensor {
                inner: Arc::new(Mutex::new(
                    (self
                        .inner
                        .lock()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .deref()
                        / t.inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                )),
            }),
            TensorOrScalar::Scalar(scalar) => Ok(PyTensor {
                inner: Arc::new(Mutex::new(
                    (self
                        .inner
                        .lock()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .deref()
                        / scalar)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                )),
            }),
        }
    }

    fn __radd__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        // Addition is commutative: scalar + tensor = tensor + scalar
        self.__add__(other)
    }

    fn __rsub__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        // scalar - tensor (NOT commutative)
        match other {
            TensorOrScalar::Tensor(t) => Ok(PyTensor {
                inner: Arc::new(Mutex::new(
                    (t.inner
                        .lock()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .deref()
                        - self
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                )),
            }),
            TensorOrScalar::Scalar(scalar) => Ok(PyTensor {
                inner: Arc::new(Mutex::new(
                    scalar
                        - self
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                )),
            }),
        }
    }

    fn __rmul__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        // Multiplication is commutative: scalar * tensor = tensor * scalar
        self.__mul__(other)
    }

    fn __rtruediv__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        // scalar / tensor (NOT commutative)
        match other {
            TensorOrScalar::Tensor(t) => Ok(PyTensor {
                inner: Arc::new(Mutex::new(
                    (t.inner
                        .lock()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .deref()
                        / self
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                )),
            }),
            TensorOrScalar::Scalar(scalar) => Ok(PyTensor {
                inner: Arc::new(Mutex::new(
                    scalar
                        - self
                            .inner
                            .lock()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                            .deref(),
                )),
            }),
        }
    }

    fn __neg__(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(
                self.inner
                    .lock()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .deref()
                    * -1.0,
            )),
        })
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<f32> {
        let indices: Vec<usize> = if let Ok(single) = index.extract::<usize>() {
            vec![single]
        } else if let Ok(tuple) = index.extract::<Vec<usize>>() {
            tuple
        } else {
            return Err(PyRuntimeError::new_err(
                "Index must be int or tuple of ints".to_string(),
            ));
        };

        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .deref()
            .data
            .get(indices.as_slice())
            .copied()
            .ok_or(PyRuntimeError::new_err("Index out of bounds".to_string()))
    }

    fn __setitem__(&mut self, index: &Bound<'_, PyAny>, value: f32) -> PyResult<()> {
        let indices: Vec<usize> = if let Ok(single) = index.extract::<usize>() {
            vec![single]
        } else if let Ok(tuple) = index.extract::<Vec<usize>>() {
            tuple
        } else {
            return Err(PyRuntimeError::new_err(
                "Index must be int or tuple of ints".to_string(),
            ));
        };

        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .deref_mut()
            .data
            .get_mut(indices.as_slice())
            .ok_or(PyRuntimeError::new_err("Index out of bounds".to_string()))
            .map(|item| *item = value)
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .deref()
            .length())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<TensorIter>> {
        let iter = TensorIter {
            inner: slf
                .inner
                .lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .deref()
                .data
                .clone()
                .into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "{:?}",
            self.inner
                .lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .deref()
                .data
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Tensor(shape={:?}, data={:?})",
            self.inner
                .lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .deref()
                .shape,
            self.inner
                .lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .deref()
                .data
        ))
    }

    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(copy_val) = copy {
            let copy_bool: bool = copy_val.extract()?;
            if !copy_bool {
                return Err(PyRuntimeError::new_err(
                    "Cannot avoid copy when converting PyTensor to NumPy array",
                ));
            }
        }

        let numpy_array = self.to_numpy(py)?;

        if let Some(dtype_val) = dtype {
            let numpy_mod = py.import("numpy")?;
            let kwargs = [("dtype", dtype_val)].into_py_dict(py)?;
            return numpy_mod.call_method("asarray", (numpy_array,), Some(&kwargs));
        }

        Ok(numpy_array.into_any())
    }
}
