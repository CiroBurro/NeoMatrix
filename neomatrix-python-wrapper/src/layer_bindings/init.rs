//! Python bindings for weight initialization strategies.
//!
//! Provides enum-style access to initialization methods: `Init.He`, `Init.Xavier`, `Init.Random`.
//! See `neomatrix_core::layers::init` module documentation for mathematical formulas.

use std::sync::{Arc, Mutex};

use crate::tensor_bindings::PyTensor;
use neomatrix_core::layers::init::Init;
use pyo3::prelude::*;

/// Python wrapper for weight initialization strategies.
///
/// Provides class attributes for each initialization method:
/// - `Init.He`: He initialization (√(2/n_in)) — best for ReLU networks
/// - `Init.Xavier`: Xavier/Glorot (√(2/(n_in+n_out))) — best for Sigmoid/Tanh
/// - `Init.Random`: Uniform random in specified range
///
/// # Usage
///
/// ```python
/// from neomatrix._backend import Dense, Init
///
/// # Using He initialization for ReLU network
/// layer = Dense(input_size=784, output_size=128, init=Init.He)
///
/// # Using Xavier for Sigmoid network
/// layer = Dense(input_size=128, output_size=10, init=Init.Xavier)
///
/// # Using Random with custom range
/// tensor = Init.Random.init(in_feat=100, out_feat=50, range_start=-0.5, range_end=0.5)
/// ```
#[pyclass(name = "Init")]
#[derive(Clone, Debug)]
pub struct PyInit {
    pub inner: Init,
}

#[pymethods]
impl PyInit {
    /// He initialization — recommended for ReLU activations.
    ///
    /// Variance-scaled initialization: weights ~ N(0, √(2/n_in))
    /// Prevents vanishing gradients in deep ReLU networks.
    #[classattr]
    #[allow(non_snake_case)]
    pub fn He() -> Self {
        PyInit { inner: Init::He }
    }

    /// Xavier/Glorot initialization — recommended for Sigmoid/Tanh activations.
    ///
    /// Variance-scaled initialization: weights ~ N(0, √(2/(n_in+n_out)))
    /// Maintains gradient variance through layers with saturating activations.
    #[classattr]
    #[allow(non_snake_case)]
    pub fn Xavier() -> Self {
        PyInit {
            inner: Init::Xavier,
        }
    }

    /// Uniform random initialization in a user-specified range.
    ///
    /// Weights ~ U(range_start, range_end). Requires explicit range parameters
    /// when calling `init()`.
    #[classattr]
    #[allow(non_snake_case)]
    pub fn Random() -> Self {
        PyInit {
            inner: Init::Random,
        }
    }

    /// Initialize a weight tensor with the selected strategy.
    ///
    /// # Arguments
    ///
    /// * `in_feat` - Number of input features (fan-in)
    /// * `out_feat` - Number of output features (fan-out)
    /// * `range_start` - Start of uniform range (required for Init.Random, ignored otherwise)
    /// * `range_end` - End of uniform range (required for Init.Random, ignored otherwise)
    ///
    /// # Returns
    ///
    /// A tensor of shape `(in_feat, out_feat)` initialized according to the strategy.
    ///
    /// # Example
    ///
    /// ```python
    /// # He initialization (range_start/range_end ignored)
    /// weights = Init.He.init(in_feat=784, out_feat=128)
    ///
    /// # Random initialization (requires both range parameters)
    /// weights = Init.Random.init(in_feat=100, out_feat=50, range_start=-1.0, range_end=1.0)
    /// ```
    #[pyo3(signature = (in_feat, out_feat, range_start=None, range_end=None))]
    pub fn init(
        &self,
        in_feat: usize,
        out_feat: usize,
        range_start: Option<f32>,
        range_end: Option<f32>,
    ) -> PyTensor {
        // Convert Option<f32> pair to Option<Range<f32>>
        let rg = match (range_start, range_end) {
            (Some(start), Some(end)) => Some(start..end),
            _ => None,
        };
        PyTensor {
            inner: Arc::new(Mutex::new(self.inner.init(in_feat, out_feat, rg))),
        }
    }

    /// Python repr: returns 'Init.He', 'Init.Xavier', or 'Init.Random'.
    fn __repr__(&self) -> &'static str {
        match self.inner {
            Init::He => "Init.He",
            Init::Xavier => "Init.Xavier",
            Init::Random => "Init.Random",
        }
    }
}
