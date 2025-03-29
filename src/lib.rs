pub mod structures;
pub mod functions;
pub mod utils;

use pyo3::prelude::*;
use structures::{
    tensor::Tensor,
    layer::Layer,
};
use functions::activation::Activation;
use utils::*;



/// A Python module implemented in Rust.
#[pymodule]
fn neomatrix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    m.add_class::<Layer>()?;
    m.add_class::<Activation>()?;
    Ok(())
}
