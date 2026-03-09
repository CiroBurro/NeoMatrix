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
    pub inner: Tensor,
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
        let inner = Tensor::new(shape, content);
        Ok(PyTensor {
            inner: inner.map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (shape))]
    pub fn zeros(shape: Vec<usize>) -> Self {
        let inner = Tensor::zeros(shape);
        PyTensor { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (shape, start=-1.0, end=1.0))]
    pub fn random(shape: Vec<usize>, start: Option<f32>, end: Option<f32>) -> Self {
        let start = start.unwrap_or(-1.0);
        let end = end.unwrap_or(1.0);
        let inner = Tensor::random(shape, start..end);
        PyTensor { inner }
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
        Ok(PyTensor { inner })
    }

    /// Getter method for the data field
    /// It converts the ndarray dynamic array into a numpy PyArrayDyn object for python
    #[getter]
    fn get_data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f32>> {
        self.inner.data.to_pyarray(py)
    }

    /// Setter method for the data field
    /// It allows to set the data field from python with a numpy array
    #[setter]
    fn set_data<'py>(&mut self, arr: PyReadonlyArrayDyn<'py, f32>) -> PyResult<()> {
        let owned = arr.as_array().to_owned();
        self.inner.dimension = owned.ndim();
        self.inner.shape = owned.shape().to_vec();
        self.inner.data = owned;
        Ok(())
    }

    pub fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f32>> {
        self.inner.data.to_pyarray(py)
    }

    #[getter]
    fn shape(&self) -> &Vec<usize> {
        &self.inner.shape
    }

    #[getter]
    fn ndim(&self) -> &usize {
        &self.inner.dimension
    }

    #[pyo3(signature = (t))]
    pub fn dot(&self, t: &PyTensor) -> PyResult<PyTensor> {
        let inner = self.inner.dot(&t.inner);
        Ok(PyTensor {
            inner: inner.map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    pub fn transpose(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self
                .inner
                .transpose()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    pub fn transpose_inplace(&mut self) -> PyResult<()> {
        self.inner
            .transpose_inplace()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (shape))]
    pub fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self
                .inner
                .reshape(shape)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    pub fn reshape_inplace(&mut self, shape: Vec<usize>) -> PyResult<()> {
        self.inner
            .reshape_inplace(shape)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn flatten(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.flatten(),
        })
    }

    pub fn flatten_inplace(&mut self) -> PyResult<()> {
        Ok(self.inner.flatten_inplace())
    }

    #[pyo3(signature = (t, axis))]
    pub fn push(&mut self, t: &PyTensor, axis: usize) -> PyResult<()> {
        self.inner
            .push(&t.inner, axis)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (tensors, axis))]
    pub fn cat_inplace(&mut self, tensors: Vec<PyRef<'_, PyTensor>>, axis: usize) -> PyResult<()> {
        let inner_tensors: Vec<Tensor> = tensors.iter().map(|t| t.inner.clone()).collect();
        self.inner = self
            .inner
            .cat_inplace(inner_tensors, axis)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    #[pyo3(signature = (tensors, axis))]
    pub fn cat(tensors: Vec<PyRef<'_, PyTensor>>, axis: usize) -> PyResult<PyTensor> {
        let inner_tensors: Vec<Tensor> = tensors.iter().map(|t| t.inner.clone()).collect();
        Ok(PyTensor {
            inner: Tensor::cat(inner_tensors, axis)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    #[pyo3(signature = (t))]
    pub fn push_row(&mut self, t: &PyTensor) -> PyResult<()> {
        self.inner
            .push_row(&t.inner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (t))]
    pub fn push_column(&mut self, t: &PyTensor) -> PyResult<()> {
        self.inner
            .push_column(&t.inner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __add__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        match other {
            TensorOrScalar::Tensor(t) => Ok(PyTensor {
                inner: (&self.inner + &t.inner)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            }),
            TensorOrScalar::Scalar(scalar) => Ok(PyTensor {
                inner: &self.inner + scalar,
            }),
        }
    }

    fn __sub__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        match other {
            TensorOrScalar::Tensor(t) => Ok(PyTensor {
                inner: (&self.inner - &t.inner)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            }),
            TensorOrScalar::Scalar(scalar) => Ok(PyTensor {
                inner: &self.inner - scalar,
            }),
        }
    }

    fn __mul__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        match other {
            TensorOrScalar::Tensor(t) => Ok(PyTensor {
                inner: (&self.inner * &t.inner)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            }),
            TensorOrScalar::Scalar(scalar) => Ok(PyTensor {
                inner: &self.inner * scalar,
            }),
        }
    }

    fn __truediv__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        match other {
            TensorOrScalar::Tensor(t) => Ok(PyTensor {
                inner: (&self.inner / &t.inner)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            }),
            TensorOrScalar::Scalar(scalar) => Ok(PyTensor {
                inner: (&self.inner / scalar)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
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
                inner: (&t.inner - &self.inner)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            }),
            TensorOrScalar::Scalar(scalar) => Ok(PyTensor {
                inner: scalar - &self.inner,
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
                inner: (&t.inner / &self.inner)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            }),
            TensorOrScalar::Scalar(scalar) => Ok(PyTensor {
                inner: (scalar / &self.inner)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            }),
        }
    }

    fn __neg__(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: &self.inner * -1.0,
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
            .data
            .get_mut(indices.as_slice())
            .ok_or(PyRuntimeError::new_err("Index out of bounds".to_string()))
            .map(|item| *item = value)
    }

    fn __len__(&self) -> usize {
        self.inner.length()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<TensorIter>> {
        let iter = TensorIter {
            inner: slf.inner.data.clone().into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner.data)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, data={:?})",
            self.inner.shape, self.inner.data
        )
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

        let numpy_array = self.to_numpy(py);

        if let Some(dtype_val) = dtype {
            let numpy_mod = py.import("numpy")?;
            let kwargs = [("dtype", dtype_val)].into_py_dict(py)?;
            return numpy_mod.call_method("asarray", (numpy_array,), Some(&kwargs));
        }

        Ok(numpy_array.into_any())
    }
}
