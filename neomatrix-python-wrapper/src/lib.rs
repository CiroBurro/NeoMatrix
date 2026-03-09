use pyo3::prelude::*;

mod tensor;
mod tensor_iter;

use tensor::PyTensor;

#[pymodule]
fn _backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    Ok(())
}
