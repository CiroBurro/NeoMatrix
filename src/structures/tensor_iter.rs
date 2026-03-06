//! Iterator implementation for Tensor elements.
//!
//! Provides Python-compatible iteration over tensor data in row-major (C) order.
//! Used to support Python's `for x in tensor:` syntax.
//!
//! # Example
//!
//! ```python
//! from neomatrix.core import Tensor
//!
//! t = Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
//! for value in t:
//!     print(value)  # Prints 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
//! ```

use ndarray::IxDyn;
use pyo3::{pyclass, pymethods, PyRef, PyRefMut};

#[pyclass(unsendable)]
pub struct TensorIter {
    pub inner: ndarray::iter::IntoIter<f32, IxDyn>,
}

#[pymethods]
impl TensorIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<f32> {
        slf.inner.next()
    }
}
