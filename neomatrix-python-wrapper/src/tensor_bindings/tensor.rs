//! Python bindings for the Tensor data structure.
//!
//! This module provides Python-accessible wrappers around the Rust `Tensor` type from
//! `neomatrix_core`, enabling seamless NumPy interoperability and native Python operators.
//!
//! # Features
//!
//! - **Full NumPy compatibility**: Implements `__array__` protocol for zero-copy conversion
//! - **Thread-safe**: Uses `Arc<Mutex<Tensor>>` for safe shared ownership in multi-threaded contexts
//! - **Pythonic operators**: Supports `+`, `-`, `*`, `/` with both tensors and scalars
//! - **Rich API**: Construction, manipulation (reshape/transpose/flatten), concatenation, indexing
//! - **Automatic broadcasting**: Scalar operations broadcast element-wise
//!
//! # Core Types
//!
//! - **`PyTensor`**: Python-exposed tensor wrapper
//! - **`TensorOrScalar`**: Internal enum for handling `Tensor + Tensor` and `Tensor + f32` operations
//! - **`TensorIter`**: Iterator for `for x in tensor` loops
//!
//! # NumPy Interoperability
//!
//! PyTensor automatically converts to/from NumPy arrays:
//! ```python
//! import numpy as np
//! from neomatrix._backend import Tensor
//!
//! # NumPy → Tensor
//! arr = np.array([[1.0, 2.0], [3.0, 4.0]])
//! t = Tensor.from_numpy(arr)
//!
//! # Tensor → NumPy (zero-copy when possible)
//! arr_out = t.to_numpy()
//! arr_out = np.array(t)  # Uses __array__ protocol
//! ```
//!
//! # Python API Examples
//!
//! ```python
//! from neomatrix._backend import Tensor
//!
//! # Construction
//! t1 = Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
//! t2 = Tensor.zeros([3, 3])
//! t3 = Tensor.random([4, 4], start=-1.0, end=1.0)
//!
//! # Operators
//! result = (t1 + 5.0) * t1 - t1.dot(t1.transpose())
//!
//! # Shape manipulation
//! t1.reshape([3, 2])
//! t1.flatten()
//! t1.transpose()
//!
//! # Concatenation
//! t_row = Tensor.zeros([1, 3])
//! t1.push_row(t_row)
//!
//! # Indexing
//! t1[0, 1]  # Get element
//! t1[0, 1] = 10.0  # Set element
//! ```
//!
//! # Thread Safety
//!
//! All operations acquire a mutex lock before accessing the underlying tensor data.
//! This ensures correctness in multi-threaded Python environments but may introduce
//! lock contention under heavy concurrent access.

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use neomatrix_core::tensor::Tensor;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, prelude::*};
use pyo3::Bound;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

use crate::tensor_bindings::tensor_iter::TensorIter;

/// Python-accessible Tensor wrapper.
///
/// **Internal Structure:** Uses `Arc<Mutex<Tensor>>` for thread-safe shared ownership.
///
/// **NumPy Protocol:** Implements `__array__` for automatic conversion via `np.array(tensor)`.
///
/// **Properties:**
/// - `data` (get/set): Raw NumPy array view
/// - `shape` (get): Dimensions as tuple
/// - `ndim` (get): Number of dimensions
///
/// **Protocols:**
/// - Arithmetic: `+`, `-`, `*`, `/` (element-wise or broadcasting)
/// - Indexing: `tensor[i, j]`, `tensor[i, j] = value`
/// - Iteration: `for x in tensor` (flattened iteration)
/// - Repr: `print(tensor)` shows shape and data
#[pyclass(name = "Tensor")]
#[derive(Clone, Debug)]
pub struct PyTensor {
    pub inner: Arc<Mutex<Tensor>>,
}

/// Internal enum for handling `Tensor OP Tensor` and `Tensor OP Scalar` operations.
///
/// Enables Python expressions like:
/// - `tensor + tensor` (Tensor variant)
/// - `tensor + 5.0` (Scalar variant)
#[derive(FromPyObject)]
enum TensorOrScalar<'py> {
    Tensor(PyRef<'py, PyTensor>),
    Scalar(f32),
}

#[pymethods]
impl PyTensor {
    /// Creates a new tensor from shape and flattened data.
    ///
    /// # Arguments
    /// * `shape` - Dimensions (e.g., `[2, 3]` for 2×3 matrix)
    /// * `content` - Flattened data in row-major order
    ///
    /// # Returns
    /// New `Tensor` instance
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if `len(content) != product(shape)`.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])  # [[1, 2], [3, 4]]
    /// ```
    #[pyo3(signature = (shape, content))]
    #[new]
    pub fn new(shape: Vec<usize>, content: Vec<f32>) -> PyResult<PyTensor> {
        let inner =
            Tensor::new(shape, content).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Creates a tensor filled with zeros.
    ///
    /// # Arguments
    /// * `shape` - Dimensions
    ///
    /// # Returns
    /// Tensor with all elements initialized to 0.0
    ///
    /// # Example
    /// ```python
    /// t = Tensor.zeros([3, 3])  # 3×3 zero matrix
    /// ```
    #[staticmethod]
    #[pyo3(signature = (shape))]
    pub fn zeros(shape: Vec<usize>) -> PyResult<Self> {
        let inner = Tensor::zeros(shape);
        Ok(PyTensor {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Creates a tensor filled with random values.
    ///
    /// # Arguments
    /// * `shape` - Dimensions
    /// * `start` - Lower bound (default: -1.0)
    /// * `end` - Upper bound (default: 1.0)
    ///
    /// # Returns
    /// Tensor with uniformly distributed random values in [start, end)
    ///
    /// # Example
    /// ```python
    /// t = Tensor.random([2, 2], start=0.0, end=1.0)  # Random in [0, 1)
    /// ```
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

    /// Creates a tensor from a NumPy array.
    ///
    /// # Arguments
    /// * `arr` - NumPy array (any shape, must be f32 dtype)
    ///
    /// # Returns
    /// Tensor containing a copy of the NumPy data
    ///
    /// # Example
    /// ```python
    /// import numpy as np
    /// arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    /// t = Tensor.from_numpy(arr)
    /// ```
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

    /// Gets the tensor data as a NumPy array.
    ///
    /// Returns a **view** (zero-copy when possible) of the underlying tensor data.
    ///
    /// # Returns
    /// NumPy array with same shape and data
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if mutex lock fails.
    #[getter]
    fn get_data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .data
            .to_pyarray(py))
    }

    /// Sets the tensor data from a NumPy array.
    ///
    /// Replaces the underlying tensor with a copy of the NumPy array's data.
    ///
    /// # Arguments
    /// * `arr` - NumPy array (must match or will reshape tensor)
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if mutex lock fails.
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

    /// Converts the tensor to a NumPy array.
    ///
    /// Identical to `get_data()` but callable as method: `tensor.to_numpy()`.
    ///
    /// # Returns
    /// NumPy array view of tensor data
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if mutex lock fails.
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .data
            .to_pyarray(py))
    }

    /// Gets the tensor shape.
    ///
    /// # Returns
    /// Shape as list of dimensions (e.g., `[2, 3]`)
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if mutex lock fails.
    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .shape
            .clone())
    }

    /// Gets the number of dimensions.
    ///
    /// # Returns
    /// Number of axes (0 for scalar, 1 for vector, 2 for matrix, etc.)
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if mutex lock fails.
    #[getter]
    fn ndim(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .dimension)
    }

    /// Matrix multiplication / dot product.
    ///
    /// Computes tensor-tensor product following standard linear algebra rules:
    /// - 1D × 1D: Inner product (scalar)
    /// - 2D × 1D: Matrix-vector product
    /// - 2D × 2D: Matrix multiplication
    /// - ND × MD: Generalized tensor contraction on last axis of self and second-to-last of other
    ///
    /// # Arguments
    /// * `t` - Right operand tensor
    ///
    /// # Returns
    /// Result tensor with appropriate shape
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if shapes incompatible or mutex lock fails.
    ///
    /// # Example
    /// ```python
    /// A = Tensor([2, 3], [...])
    /// B = Tensor([3, 4], [...])
    /// C = A.dot(B)  # Shape: [2, 4]
    /// ```
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

    /// Transposes the tensor (returns a new tensor).
    ///
    /// Swaps the last two axes. For 2D matrices, this is standard transpose (rows ↔ columns).
    ///
    /// # Returns
    /// Transposed tensor
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensor has dimension < 2 or mutex lock fails.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// t_T = t.transpose()  # Shape: [3, 2]
    /// ```
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

    /// Transposes the tensor in-place (modifies self).
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if tensor has dimension < 2 or mutex lock fails.
    pub fn transpose_inplace(&mut self) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .deref_mut()
            .transpose_inplace()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Reshapes the tensor to a new shape (returns a new tensor).
    ///
    /// # Arguments
    /// * `shape` - New dimensions (must have same total elements)
    ///
    /// # Returns
    /// Reshaped tensor
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if `product(new_shape) != product(old_shape)` or mutex lock fails.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// t_new = t.reshape([3, 2])  # Same data, different shape
    /// ```
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

    /// Reshape the tensor in-place to a new shape (mutates self).
    ///
    /// # Arguments
    /// * `shape` - Target shape (must have same total element count)
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if product(shape) ≠ current element count or mutex lock fails.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([6], [1, 2, 3, 4, 5, 6])
    /// t.reshape_inplace([2, 3])  # Mutates t to shape [2, 3]
    /// ```
    pub fn reshape_inplace(&mut self, shape: Vec<usize>) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .deref_mut()
            .reshape_inplace(shape)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Flatten the tensor to 1D (returns new tensor).
    ///
    /// # Returns
    /// New 1D tensor containing all elements in row-major order.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// t_flat = t.flatten()  # Shape: [6], data: [1, 2, 3, 4, 5, 6]
    /// ```
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

    /// Flatten the tensor to 1D in-place (mutates self).
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// t.flatten_inplace()  # Mutates t to shape [6]
    /// ```
    pub fn flatten_inplace(&mut self) -> PyResult<()> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .deref_mut()
            .flatten_inplace())
    }

    /// Concatenate another tensor along the specified axis (mutates self).
    ///
    /// # Arguments
    /// * `t` - Tensor to append
    /// * `axis` - Axis along which to concatenate (0 = rows, 1 = columns for 2D)
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if shapes are incompatible or mutex lock fails.
    ///
    /// # Example
    /// ```python
    /// t1 = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// t2 = Tensor([2, 3], [7, 8, 9, 10, 11, 12])
    /// t1.push(t2, axis=0)  # Shape becomes [4, 3]
    /// ```
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

    /// Concatenate multiple tensors along the specified axis (static method).
    ///
    /// # Arguments
    /// * `tensors` - List of tensors to concatenate
    /// * `axis` - Axis along which to concatenate
    ///
    /// # Returns
    /// New tensor containing all input tensors concatenated.
    ///
    /// # Example
    /// ```python
    /// t1 = Tensor([2, 2], [1, 2, 3, 4])
    /// t2 = Tensor([2, 2], [5, 6, 7, 8])
    /// t3 = Tensor.cat([t1, t2], axis=0)  # Shape: [4, 2]
    /// ```
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

    /// Append a row to the tensor (mutates self). Promotes 1D→2D if needed.
    ///
    /// # Arguments
    /// * `t` - 1D tensor to append as a new row
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if t is not 1D or has incompatible length.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// row = Tensor([3], [7, 8, 9])
    /// t.push_row(row)  # Shape becomes [3, 3]
    /// ```
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

    /// Append a column to the tensor (mutates self). Promotes 1D→2D if needed.
    ///
    /// # Arguments
    /// * `t` - 1D tensor to append as a new column
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if t is not 1D or has incompatible length.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([3, 2], [1, 2, 3, 4, 5, 6])
    /// col = Tensor([3], [7, 8, 9])
    /// t.push_column(col)  # Shape becomes [3, 3]
    /// ```
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

    /// Python `+` operator: element-wise addition or scalar broadcast.
    ///
    /// Supports `tensor + tensor` and `tensor + scalar`.
    ///
    /// # Example
    /// ```python
    /// t1 = Tensor([2, 2], [1, 2, 3, 4])
    /// t2 = Tensor([2, 2], [5, 6, 7, 8])
    /// result = t1 + t2  # Element-wise: [6, 8, 10, 12]
    /// result = t1 + 10  # Broadcast: [11, 12, 13, 14]
    /// ```
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

    /// Python `-` operator: element-wise subtraction or scalar broadcast.
    ///
    /// Supports `tensor - tensor` and `tensor - scalar`.
    ///
    /// # Example
    /// ```python
    /// t1 = Tensor([2, 2], [10, 20, 30, 40])
    /// t2 = Tensor([2, 2], [1, 2, 3, 4])
    /// result = t1 - t2  # Element-wise: [9, 18, 27, 36]
    /// result = t1 - 5   # Broadcast: [5, 15, 25, 35]
    /// ```
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

    /// Python `*` operator: element-wise multiplication or scalar broadcast.
    ///
    /// Supports `tensor * tensor` and `tensor * scalar`.
    /// For matrix multiplication, use `dot()` method.
    ///
    /// # Example
    /// ```python
    /// t1 = Tensor([2, 2], [1, 2, 3, 4])
    /// t2 = Tensor([2, 2], [2, 3, 4, 5])
    /// result = t1 * t2  # Element-wise: [2, 6, 12, 20]
    /// result = t1 * 10  # Broadcast: [10, 20, 30, 40]
    /// ```
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

    /// Python `/` operator: element-wise division or scalar broadcast.
    ///
    /// Supports `tensor / tensor` and `tensor / scalar`.
    ///
    /// # Example
    /// ```python
    /// t1 = Tensor([2, 2], [10, 20, 30, 40])
    /// t2 = Tensor([2, 2], [2, 4, 5, 8])
    /// result = t1 / t2  # Element-wise: [5, 5, 6, 5]
    /// result = t1 / 10  # Broadcast: [1, 2, 3, 4]
    /// ```
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

    /// Reflected `+` operator (commutative): `scalar + tensor`.
    fn __radd__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        // Addition is commutative: scalar + tensor = tensor + scalar
        self.__add__(other)
    }

    /// Reflected `-` operator (non-commutative): `scalar - tensor`.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 2], [1, 2, 3, 4])
    /// result = 10 - t  # [9, 8, 7, 6]
    /// ```
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

    /// Reflected `*` operator (commutative): `scalar * tensor`.
    fn __rmul__(&self, other: TensorOrScalar) -> PyResult<PyTensor> {
        // Multiplication is commutative: scalar * tensor = tensor * scalar
        self.__mul__(other)
    }

    /// Reflected `/` operator (non-commutative): `scalar / tensor`.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 2], [2, 4, 5, 10])
    /// result = 100 / t  # [50, 25, 20, 10]
    /// ```
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

    /// Unary negation operator: `-tensor`.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 2], [1, -2, 3, -4])
    /// result = -t  # [-1, 2, -3, 4]
    /// ```
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

    /// Python indexing operator (read): `tensor[i]` or `tensor[i, j, ...]`.
    ///
    /// Supports both flat indexing (single int) and multi-dimensional indexing (tuple of ints).
    ///
    /// # Returns
    /// The element value at the specified index (f32).
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if index is out of bounds or has wrong type.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// val = t[1, 2]  # Returns 6.0
    /// val = t[0]     # Flat index: returns 1.0
    /// ```
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

    /// Python indexing operator (write): `tensor[i] = value` or `tensor[i, j, ...] = value`.
    ///
    /// Supports both flat indexing (single int) and multi-dimensional indexing (tuple of ints).
    ///
    /// # Arguments
    /// * `index` - Index (int or tuple of ints)
    /// * `value` - New value to assign (f32)
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if index is out of bounds or has wrong type.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// t[1, 2] = 100.0  # Modifies element at (1, 2)
    /// t[0] = 50.0      # Flat index: modifies first element
    /// ```
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

    /// Python `len()` function: returns total number of elements.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// length = len(t)  # Returns 6
    /// ```
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

    /// Python `repr()` function: string representation for debugging.
    ///
    /// # Example
    /// ```python
    /// t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    /// print(repr(t))  # "Tensor(dimension=2, shape=[2, 3])"
    /// ```
    fn __repr__(&self) -> PyResult<String> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(format!(
            "Tensor(shape={:?}, data={:?})",
            inner.deref().shape,
            inner.deref().data
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
