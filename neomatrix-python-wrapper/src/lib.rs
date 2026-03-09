use pyo3::prelude::*;

mod layer_bindings;
mod losses_bindings;
mod tensor_bindings;

use layer_bindings::activations::PyReLU;
use layer_bindings::activations::PySigmoid;
use layer_bindings::activations::PySoftmax;
use layer_bindings::activations::PyTanh;
use layer_bindings::dense::PyDense;
use layer_bindings::init::PyInit;
use losses_bindings::{
    PyBinaryCrossEntropy, PyCategoricalCrossEntropy, PyHingeLoss, PyHuberLoss, PyMeanAbsoluteError,
    PyMeanSquaredError,
};
use tensor_bindings::PyTensor;

#[pymodule]
fn _backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyDense>()?;
    m.add_class::<PyReLU>()?;
    m.add_class::<PySigmoid>()?;
    m.add_class::<PyTanh>()?;
    m.add_class::<PySoftmax>()?;
    m.add_class::<PyInit>()?;
    m.add_class::<PyMeanSquaredError>()?;
    m.add_class::<PyMeanAbsoluteError>()?;
    m.add_class::<PyBinaryCrossEntropy>()?;
    m.add_class::<PyCategoricalCrossEntropy>()?;
    m.add_class::<PyHuberLoss>()?;
    m.add_class::<PyHingeLoss>()?;
    Ok(())
}
