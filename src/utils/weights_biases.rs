//! Random weight and bias tensor initialization for neural network layers.
//!
//! Provides Python-exposed functions to generate random tensors for layer parameters.
//! Uses uniform distribution within specified range.
//!
//! # Functions
//!
//! - `random_weights(nodes_1, nodes_2, range)`: 2D tensor for layer weights
//! - `random_biases(nodes, range)`: 1D tensor for layer biases
//!
//! # Example (Python)
//!
//! ```python
//! from rustybrain import random_weights, random_biases
//!
//! W = random_weights(784, 128, (-0.5, 0.5))  # 784×128 weights
//! b = random_biases(128, (-0.1, 0.1))        # 128 biases
//! ```

use crate::structures::tensor::Tensor;
use ndarray::prelude::*;
use pyo3::pyfunction;
use rand::{rng, Rng};

#[pyfunction]
#[pyo3(signature = (nodes_1, nodes_2, range))]
pub fn random_weights(nodes_1: usize, nodes_2: usize, range: (f32, f32)) -> Tensor {
    let mut rng = rng();

    let weights = Array::from_shape_fn((nodes_1, nodes_2), |_| {
        let weight: f32 = rng.random_range(range.0..range.1);
        weight
    });
    Tensor {
        dimension: 2,
        shape: vec![nodes_1, nodes_2],
        data: weights.into_dyn(),
    }
}

#[pyfunction]
#[pyo3(signature = (nodes, range))]
pub fn random_biases(nodes: usize, range: (f32, f32)) -> Tensor {
    let mut rng = rng();

    let biases = Array::from_shape_fn(nodes, |_| {
        let bias: f32 = rng.random_range(range.0..range.1);
        bias
    });
    Tensor {
        dimension: 1,
        shape: vec![nodes],
        data: biases.into_dyn(),
    }
}
