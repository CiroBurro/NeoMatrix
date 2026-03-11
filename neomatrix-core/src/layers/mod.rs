//! Neural network layer implementations and abstractions.
//!
//! This module provides the foundational building blocks for constructing neural networks:
//!
//! - **[`Layer`]**: Core trait defining the forward/backward pass interface
//! - **[`dense`]**: Fully-connected (dense) layer with learnable weights and biases
//! - **[`activations`]**: Activation layer wrappers (ReLU, Sigmoid, Tanh, Softmax) with backprop caching
//! - **[`init`]**: Weight initialization strategies (Random, Xavier, He)
//!
//! # Layer Trait
//!
//! All layers implement the [`Layer`] trait, which provides:
//! - **Forward pass**: Transform input tensor to output tensor
//! - **Backward pass**: Compute gradients with respect to inputs
//! - **Parameter access**: Retrieve learnable weights and their gradients (for optimization)
//!
//! # Example
//!
//! ```rust,ignore
//! use neomatrix_core::layers::{dense::Dense, init::Init, Layer};
//! use neomatrix_core::tensor::Tensor;
//!
//! // Create a dense layer: 3 inputs → 5 outputs, Xavier initialization
//! let mut layer = Dense::new(3, 5, Some(Init::Xavier), None).unwrap();
//!
//! // Forward pass
//! let input = Tensor::random(&[1, 3], -1.0..1.0).unwrap();
//! let output = layer.forward(&input, true).unwrap();
//!
//! // Backward pass (after loss computation)
//! let grad_output = Tensor::ones(&[1, 5]).unwrap();
//! let grad_input = layer.backward(&grad_output).unwrap();
//!
//! // Access weights and gradients for optimization
//! if let Some(params) = layer.get_params_and_grads() {
//!     for (weight, gradient) in params {
//!         // Update weights using gradient descent
//!     }
//! }
//! ```

pub mod activations;
pub mod dense;
pub mod init;

use crate::errors::LayerError;
use crate::optimizers::ParametersRef;
use crate::tensor::Tensor;

/// Core trait for all neural network layers.
///
/// This trait defines the standard interface for building composable layers in a neural network.
/// Each layer must implement forward propagation, backpropagation, and optionally expose
/// learnable parameters.
///
/// # Implementation Requirements
///
/// - **[`forward`](Layer::forward)**: Must compute the layer's output given an input tensor.
///   The `training` flag allows layers to behave differently during training vs. inference
///   (e.g., dropout, batch normalization).
///
/// - **[`backward`](Layer::backward)**: Must compute the gradient with respect to the layer's input,
///   given the gradient with respect to the output. For layers with learnable parameters,
///   this method should also compute and cache parameter gradients internally.
///
/// - **[`get_params_and_grads`](Layer::get_params_and_grads)**: Returns mutable references to
///   weights paired with their gradients. This allows optimizers to update parameters.
///   Layers without learnable parameters (e.g., activation layers) return `None`.
///
/// # Design Notes
///
/// The trait requires `&mut self` for all methods to allow internal state management:
/// - Caching intermediate values during forward pass for use in backward pass
/// - Accumulating gradients for learnable parameters
/// - Maintaining running statistics (e.g., batch normalization moving averages)
pub trait Layer {
    /// Performs the forward pass through the layer.
    ///
    /// Transforms the input tensor according to the layer's computation (e.g., linear transformation,
    /// activation function). During training, layers may cache intermediate values needed for
    /// backpropagation.
    ///
    /// # Parameters
    ///
    /// - `input`: Input tensor with shape `[batch_size, input_features]` (or layer-specific shape)
    /// - `training`: Whether the model is in training mode. Set to `true` during training,
    ///   `false` during inference. Some layers (e.g., dropout) behave differently based on this flag.
    ///
    /// # Returns
    ///
    /// - `Ok(Tensor)`: Output tensor with shape determined by the layer type
    /// - `Err(LayerError)`: If computation fails (e.g., shape mismatch, numerical error)
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError>;

    /// Performs the backward pass through the layer (backpropagation).
    ///
    /// Computes the gradient of the loss with respect to the layer's input, given the gradient
    /// with respect to the output. For layers with learnable parameters (e.g., Dense layer),
    /// this method also computes and internally stores the gradients with respect to those parameters.
    ///
    /// # Parameters
    ///
    /// - `output_gradient`: Gradient of the loss with respect to this layer's output,
    ///   typically propagated from the next layer. Shape: `[batch_size, output_features]`
    ///
    /// # Returns
    ///
    /// - `Ok(Tensor)`: Gradient of the loss with respect to this layer's input.
    ///   Shape: `[batch_size, input_features]`
    /// - `Err(LayerError)`: If computation fails (e.g., cached values missing, shape mismatch)
    ///
    /// # Implementation Note
    ///
    /// Layers must cache necessary values from the forward pass (e.g., input, output, or
    /// intermediate activations) to compute gradients correctly.
    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError>;

    /// Retrieves mutable references to learnable parameters and their gradients.
    ///
    /// This method provides access to the layer's trainable weights and their corresponding
    /// gradients for use by optimization algorithms (SGD, Adam, etc.). Each tuple contains:
    /// - `&mut Tensor`: Mutable reference to the parameter (weight or bias)
    /// - `&Tensor`: Immutable reference to the gradient computed during backward pass
    ///
    /// # Returns
    ///
    /// - `Some(Vec<(&mut Tensor, &Tensor)>)`: For layers with learnable parameters (e.g., Dense layer)
    /// - `None`: For layers without learnable parameters (e.g., ReLU, Softmax)
    ///
    /// # Example Use Case
    ///
    /// ```rust,ignore
    /// // Gradient descent update
    /// if let Some(params) = layer.get_params_and_grads() {
    ///     for (weight, gradient) in params {
    ///         *weight = weight - &(gradient * learning_rate);
    ///     }
    /// }
    /// ```
    fn get_parameters(&mut self) -> Option<ParametersRef> {
        None
    }
}
