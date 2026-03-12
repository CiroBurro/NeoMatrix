//! Python bindings for tensor operations.
//!
//! This module provides the `PyTensor` class with full NumPy interoperability and
//! iterator support for seamless integration with Python's scientific computing ecosystem.

mod tensor;
mod tensor_iter;

pub use tensor::PyTensor;
