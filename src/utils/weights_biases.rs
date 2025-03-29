/// This module contains functions to generate random weights and biases for a neural network layer.

/// Necessary imports
use rand::{rng, Rng};
use ndarray::prelude::*;
use crate::structures::tensor::Tensor;

/// Function to generate random weights
/// 
/// Parameters:
/// - nodes_1: The number of nodes in the previous layer
/// - nodes_2: The number of nodes in the current layer
pub fn random_weights(nodes_1: usize, nodes_2: usize) -> Tensor{
    let mut rng = rng();

    let weights = Array::from_shape_fn((nodes_1, nodes_2), |_| {
        let weight: f64 = rng.random_range(-1.0..1.0);
        weight
    });
    Tensor { dimension: 2, shape: vec![nodes_1, nodes_2], data: weights.into_dyn() }
}


/// Function to generate random biases
///
/// Parameters:
/// - nodes: The number of nodes in the current layer
pub fn random_biases(nodes: usize) -> Tensor{
    let mut rng = rng();

    let biases = Array::from_shape_fn(nodes, |_| {
        let bias: f64 = rng.random_range(-1.0..1.0);
        bias
    });
    Tensor { dimension: 1, shape: vec![nodes], data: biases.into_dyn() }
}