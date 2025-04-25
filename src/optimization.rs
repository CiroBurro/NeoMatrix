use pyo3::pyclass;

#[pyclass]
pub enum Optimizer {
    BatchGD,
    SGD,
    MiniBatchSGD,
}

