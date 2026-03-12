//! Iterator implementation for PyTensor to enable `for` loops in Python.

use ndarray::IxDyn;
use pyo3::{pyclass, pymethods, PyRef, PyRefMut};

/// Python iterator for tensor elements.
///
/// Enables iteration over tensor elements in Python: `for elem in tensor: ...`
/// Iterates in row-major (C-contiguous) order, flattening multi-dimensional tensors.
///
/// # Example
///
/// ```python
/// tensor = Tensor.random([2, 3])
/// for elem in tensor:
///     print(elem)  # Prints all 6 elements in row-major order
/// ```
#[pyclass(unsendable)]
pub struct TensorIter {
    pub inner: ndarray::iter::IntoIter<f32, IxDyn>,
}

#[pymethods]
impl TensorIter {
    /// Return self for Python iterator protocol.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Return the next element, or None when exhausted.
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<f32> {
        slf.inner.next()
    }
}
