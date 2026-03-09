use crate::tensor_bindings::PyTensor;
use neomatrix_core::layers::init::Init;
use pyo3::prelude::*;

/// Python-facing wrapper for the `Init` enum.
/// Used like a Python enum: `Init.He`, `Init.Xavier`, `Init.Random`
#[pyclass(name = "Init")]
#[derive(Clone, Debug)]
pub struct PyInit {
    pub inner: Init,
}

#[pymethods]
impl PyInit {
    /// He initialization — recommended for ReLU activations.
    #[classattr]
    #[allow(non_snake_case)]
    pub fn He() -> Self {
        PyInit { inner: Init::He }
    }

    /// Xavier/Glorot initialization — recommended for Sigmoid/Tanh activations.
    #[classattr]
    #[allow(non_snake_case)]
    pub fn Xavier() -> Self {
        PyInit {
            inner: Init::Xavier,
        }
    }

    /// Uniform random initialization in a user-specified range.
    #[classattr]
    #[allow(non_snake_case)]
    pub fn Random() -> Self {
        PyInit {
            inner: Init::Random,
        }
    }

    pub fn init(
        &self,
        in_feat: usize,
        out_feat: usize,
        range_start: Option<f32>,
        range_end: Option<f32>,
    ) -> PyTensor {
        let rg = match (range_start, range_end) {
            (Some(start), Some(end)) => Some(start..end),
            _ => None,
        };
        PyTensor {
            inner: self.inner.init(in_feat, out_feat, rg),
        }
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            Init::He => "Init.He",
            Init::Xavier => "Init.Xavier",
            Init::Random => "Init.Random",
        }
    }
}
