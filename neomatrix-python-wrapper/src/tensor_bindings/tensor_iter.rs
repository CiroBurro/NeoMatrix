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
