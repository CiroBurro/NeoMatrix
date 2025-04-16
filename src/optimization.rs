use pyo3::{pyclass, pyfunction, pymethods};
use crate::structures::tensor::Tensor;

#[pyclass]
pub enum Optimizer {
    BatchGD,
    SDG,
    MiniBatchSGD,
}

