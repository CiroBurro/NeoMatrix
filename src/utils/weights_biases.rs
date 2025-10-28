/// This module contains functions to generate random weights and biases for a neural network layer.
/// Necessary imports
use rand::{rng, Rng};
use ndarray::prelude::*;
use pyo3::pyfunction;
use crate::structures::tensor::Tensor;

/// Function to generate random weights
/// 
/// Parameters:
/// - nodes_1: The number of nodes in the previous layer
/// - nodes_2: The number of nodes in the current layer
#[pyfunction]
#[pyo3(signature = (nodes_1, nodes_2, range))]
pub fn random_weights(nodes_1: usize, nodes_2: usize, range: (f64, f64)) -> Tensor{
    let mut rng = rng();

    let weights = Array::from_shape_fn((nodes_1, nodes_2), |_| {
        let weight: f64 = rng.random_range(range.0..range.1);
        weight
    });
    Tensor { dimension: 2, shape: vec![nodes_1, nodes_2], data: weights.into_dyn() }
}


/// Function to generate random biases
///
/// Parameters:
/// - nodes: The number of nodes in the current layer
#[pyfunction]
#[pyo3(signature = (nodes, range))]
pub fn random_biases(nodes: usize, range: (f64, f64)) -> Tensor{
    let mut rng = rng();

    let biases = Array::from_shape_fn(nodes, |_| {
        let bias: f64 = rng.random_range(range.0..range.1);
        bias
    });
    Tensor { dimension: 1, shape: vec![nodes], data: biases.into_dyn() }
}