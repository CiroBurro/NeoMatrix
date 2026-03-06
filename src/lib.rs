//! # Rustybrain - NeoMatrix Compute Backend
//!
//! High-performance tensor computation library for machine learning, written in Rust
//! with Python bindings via PyO3. Provides the computational core for the NeoMatrix
//! deep learning framework.
//!
//! ## Architecture
//!
//! - **Tensor operations**: Multi-dimensional array operations with f32 precision
//! - **Neural network layers**: Dense layers with forward/backward propagation
//! - **Activation functions**: ReLU, Sigmoid, Tanh, Softmax, Linear
//! - **Loss functions**: MSE, MAE, Cross-entropy, Huber, Hinge
//! - **Performance**: Parallel matrix multiplication via Rayon
//!
//! ## Modules
//!
//! - [`structures`]: Core data structures (Tensor, Layer)
//! - [`functions`]: Mathematical functions (activation, cost)
//! - [`utils`]: Utility functions (weight initialization, matrix operations)
//!
//! ## Memory Model
//!
//! All tensors use f32 (32-bit floating point) for 50% memory savings and 2x SIMD
//! performance compared to f64, with ~7 decimal digits precision (sufficient for
//! deep learning).
//!
//! ## Python Bindings
//!
//! This crate compiles to a Python extension module `rustybrain` that exports:
//! - `Tensor`: Multi-dimensional array with NumPy interoperability
//! - `Layer`: Neural network layer with backpropagation
//! - `Activation`, `Cost`: Enum-based function selection
//! - Utility functions: `random_weights`, `random_biases`, `get_cost`

pub mod functions;
pub mod structures;
pub mod utils;

#[cfg(test)]
mod test;
use functions::{
    activation::Activation,
    cost::{get_cost, Cost},
};
use pyo3::prelude::*;
use structures::{layer::Layer, tensor::Tensor};
use utils::weights_biases::{random_biases, random_weights};

/// Rustybrain: the real brain of NeoMatrix
#[pymodule]
fn rustybrain(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    m.add_class::<Layer>()?;
    m.add_class::<Activation>()?;
    m.add_class::<Cost>()?;
    m.add_wrapped(wrap_pyfunction!(get_cost))?;
    m.add_wrapped(wrap_pyfunction!(random_weights))?;
    m.add_wrapped(wrap_pyfunction!(random_biases))?;
    Ok(())
}
