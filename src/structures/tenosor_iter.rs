use ndarray::IxDyn;
use pyo3::{pymethods, pyclass, PyRef, PyRefMut};

/// Struct `TensorIter`
/// 
/// It wraps an iterator over a tensor
///
/// # Fields
/// * `inner` - Actual iterator
#[pyclass(unsendable)]
pub struct TensorIter {
	pub inner: ndarray::iter::IntoIter<f64, IxDyn>
}

/// `TensorIter` struct methods
#[pymethods]
impl TensorIter {
	/// Iter method for python iterator object
	fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
		slf
	}

	/// Next method for python iterator object
	/// 
	/// # Returns
	/// * `Option<f64>` - Next value if exists
	fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<f64> {
		slf.inner.next()
	}
}
