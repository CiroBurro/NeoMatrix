//! # NeoMatrix Python Bindings
//!
//! This crate provides Python bindings for the `neomatrix-core` library using PyO3.
//! It exposes high-performance tensor operations, neural network layers, optimizers,
//! and loss functions to Python with NumPy interoperability.
//!
//! ## Module Structure
//!
//! - `tensor_bindings`: Python bindings for `Tensor` with NumPy array protocol support
//! - `layer_bindings`: Neural network layer wrappers (Dense, ReLU, Sigmoid, Tanh, Softmax)
//! - `optimizer_bindings`: Stateful optimizer implementations (GradientDescent) with parameter registration
//! - `losses_bindings`: Loss function wrappers with forward and backward passes
//!
//! ## Python API
//!
//! All classes are exposed through the `_backend` module, which is imported by the
//! high-level `neomatrix` Python package. Users should interact with the high-level API
//! rather than importing `_backend` directly.
//!
//! ## Performance
//!
//! All tensor operations are implemented in Rust using `ndarray` and parallelized with Rayon.
//! The Python wrapper adds minimal overhead through Arc<Mutex<Tensor>> for safe shared ownership.

extern crate openblas_src;

mod layer_bindings;
mod losses_bindings;
mod optimizer_bindings;
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
use pyo3::prelude::*;
use tensor_bindings::PyTensor;

use crate::optimizer_bindings::PyParametersRef;
use crate::optimizer_bindings::adagrad::PyAdagrad;
use crate::optimizer_bindings::gradient_descent::PyGD;
use crate::optimizer_bindings::momentum_gd::PyMomentumGD;

/// Python module definition for the NeoMatrix backend.
///
/// This module registers all Python-visible types (tensors, layers, optimizers, losses)
/// with the Python interpreter. It is built by maturin and imported as `neomatrix._backend`.
///
/// # Registered Types
///
/// - **Tensor**: `PyTensor` - Dynamic n-dimensional array with NumPy compatibility
/// - **Layers**: `PyDense`, `PyReLU`, `PySigmoid`, `PyTanh`, `PySoftmax`
/// - **Initialization**: `PyInit` - Weight initialization strategies (Xavier, He, Random)
/// - **Losses**: `PyMeanSquaredError`, `PyMeanAbsoluteError`, `PyBinaryCrossEntropy`,
///   `PyCategoricalCrossEntropy`, `PyHuberLoss`, `PyHingeLoss`
/// - **Optimizers**: `PyGD` (GradientDescent), `PyParametersRef` (parameter container)
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
    m.add_class::<PyGD>()?;
    m.add_class::<PyMomentumGD>()?;
    m.add_class::<PyAdagrad>()?;
    m.add_class::<PyParametersRef>()?;
    Ok(())
}
