//! Fully-connected (dense) layer implementation.
//!
//! This module provides the [`Dense`] layer, the fundamental building block of feedforward
//! neural networks. A dense layer performs an affine transformation on its input, applying
//! learned weights and biases.
//!
//! # Mathematical Operation
//!
//! Given input `X` with shape `[batch_size, in_features]`:
//! ```text
//! Y = X · W + b
//! ```
//! where:
//! - `W`: weight matrix `[in_features, out_features]`
//! - `b`: bias vector `[out_features]` (broadcasted across batch)
//! - `Y`: output `[batch_size, out_features]`
//!
//! # Backpropagation
//!
//! During backward pass, the layer computes three gradients:
//! - **Weight gradient**: `∇W = Xᵀ · ∇Y`
//! - **Bias gradient**: `∇b = sum(∇Y, axis=0)`
//! - **Input gradient**: `∇X = ∇Y · Wᵀ` (propagated to previous layer)
//!
//! # Weight Initialization
//!
//! Supports multiple initialization strategies via [`Init`]:
//! - **Xavier/Glorot**: Uniform distribution scaled by fan-in/fan-out (tanh, sigmoid)
//! - **He**: Normal distribution scaled by fan-in (ReLU activations)
//! - **LeCun**: Normal distribution scaled by fan-in (SELU activations)
//! - **Random**: Uniform distribution in specified range
//!
//! Biases are always initialized to zero.
//!
//! # Example
//!
//! ```rust,ignore
//! use neomatrix_core::layers::{dense::Dense, init::Init, Layer};
//! use neomatrix_core::tensor::Tensor;
//!
//! // Create 784 → 128 → 10 network
//! let mut hidden = Dense::new(784, 128, Some(Init::He), None);
//! let mut output = Dense::new(128, 10, Some(Init::Xavier), None);
//!
//! // Forward pass
//! let x = Tensor::random(&[32, 784], -1.0..1.0).unwrap();
//! let h = hidden.forward(&x, true).unwrap();
//! let y = output.forward(&h, true).unwrap();
//!
//! // Backward pass
//! let grad_y = Tensor::ones(&[32, 10]).unwrap();
//! let grad_h = output.backward(&grad_y).unwrap();
//! let grad_x = hidden.backward(&grad_h).unwrap();
//! ```

use std::ops::Range;

use ndarray::Axis;

use crate::errors::LayerError;
use crate::layers::init::Init;
use crate::layers::Layer;
use crate::tensor::Tensor;

/// Fully-connected (dense) neural network layer.
///
/// Performs an affine transformation: `output = input · weights + biases`
///
/// # Structure
///
/// - **Weights**: Shape `[in_features, out_features]` — learnable parameter matrix
/// - **Biases**: Shape `[out_features]` — learnable bias vector (broadcasted across batch)
/// - **Gradients**: Computed during backward pass for optimization
/// - **Input cache**: Stored during forward pass (training mode) for backpropagation
///
/// # Mathematical Operation
///
/// Given input `X` with shape `[batch_size, in_features]`:
/// - Forward: `Y = X · W + b` → shape `[batch_size, out_features]`
/// - Backward:
///   - `∇W = Xᵀ · ∇Y` (gradient w.r.t. weights)
///   - `∇b = sum(∇Y, axis=0)` (gradient w.r.t. biases)
///   - `∇X = ∇Y · Wᵀ` (gradient w.r.t. input, propagated to previous layer)
///
/// # Weight Initialization
///
/// Weights are initialized using the strategy specified in [`new`](Dense::new).
/// Biases are always initialized to zero.
///
/// # Example
///
/// ```rust,ignore
/// use neomatrix_core::layers::{dense::Dense, init::Init, Layer};
/// use neomatrix_core::tensor::Tensor;
///
/// // Create a 784 → 128 dense layer with He initialization
/// let mut layer = Dense::new(784, 128, Some(Init::He), None);
///
/// // Forward pass with batch of 32 samples
/// let input = Tensor::random(&[32, 784], -1.0..1.0).unwrap();
/// let output = layer.forward(&input, true).unwrap(); // shape: [32, 128]
/// ```
#[derive(Clone, Debug)]
pub struct Dense {
    /// Cached input tensor from forward pass, used during backpropagation.
    /// Only populated when `training=true` in forward pass.
    input_cache: Option<Tensor>,

    /// Weight matrix with shape `[in_features, out_features]`.
    /// Initialized using the strategy specified in constructor.
    weights: Tensor,

    /// Bias vector with shape `[out_features]`.
    /// Always initialized to zeros.
    biases: Tensor,

    /// Gradient of the loss with respect to weights, computed during backward pass.
    /// Shape: `[in_features, out_features]`
    weights_gradient: Option<Tensor>,

    /// Gradient of the loss with respect to biases, computed during backward pass.
    /// Shape: `[out_features]`
    biases_gradient: Option<Tensor>,
}

impl Dense {
    /// Creates a new dense layer with the specified input/output dimensions.
    ///
    /// # Parameters
    ///
    /// - `in_feat`: Number of input features (input tensor shape: `[batch_size, in_feat]`)
    /// - `out_feat`: Number of output features (output tensor shape: `[batch_size, out_feat]`)
    /// - `init`: Weight initialization strategy. Defaults to [`Init::Xavier`] if `None`.
    ///   - `Init::Random`: Random values in specified range
    ///   - `Init::Xavier`: Xavier/Glorot initialization (uniform distribution)
    ///   - `Init::He`: He initialization (optimal for ReLU activations)
    /// - `rg`: Optional range for random initialization. Only used when `init=Init::Random`.
    ///   If `None`, the initialization strategy's default range is used.
    ///
    /// # Returns
    ///
    /// A new `Dense` layer with:
    /// - Weights initialized according to `init` strategy
    /// - Biases initialized to zeros
    /// - No cached gradients or inputs
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Xavier initialization (default)
    /// let layer1 = Dense::new(128, 64, None, None);
    ///
    /// // He initialization for use with ReLU
    /// let layer2 = Dense::new(64, 32, Some(Init::He), None);
    ///
    /// // Custom random initialization
    /// let layer3 = Dense::new(32, 10, Some(Init::Random), Some(-0.5..0.5));
    /// ```
    pub fn new(
        in_feat: usize,
        out_feat: usize,
        init: Option<Init>,
        rg: Option<Range<f32>>,
    ) -> Self {
        Self {
            input_cache: None,
            weights: init
                .unwrap_or(Init::Xavier)
                .init(in_feat, out_feat, rg.clone()),
            biases: Tensor::zeros(vec![out_feat]),
            weights_gradient: None,
            biases_gradient: None,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError> {
        // Cache input for backpropagation during training
        if training {
            self.input_cache = Some(input.clone());
        }

        // Compute: output = input · weights + biases
        // Matrix multiplication followed by bias addition (broadcasted across batch dimension)
        (&input.dot(&self.weights)? + &self.biases).map_err(LayerError::from)
    }

    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError> {
        // Retrieve cached input from forward pass
        let input = self
            .input_cache
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Compute weight gradient: ∇W = Xᵀ · ∇Y
        // Shape: [in_features, batch_size] · [batch_size, out_features] = [in_features, out_features]
        self.weights_gradient = Some(input.transpose()?.dot(output_gradient)?);

        // Compute bias gradient: ∇b = sum(∇Y, axis=0)
        // Sum across batch dimension to get per-feature bias gradient
        let data = output_gradient.data.sum_axis(Axis(0)).into_dyn();
        self.biases_gradient = Some(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        });

        // Compute input gradient: ∇X = ∇Y · Wᵀ
        // This gradient is propagated to the previous layer
        output_gradient
            .dot(&self.weights.transpose()?)
            .map_err(LayerError::from)
    }

    fn get_params_and_grads(&mut self) -> Option<Vec<(&mut Tensor, &Tensor)>> {
        // Return parameters only if gradients have been computed (i.e., after backward pass)
        if self.weights_gradient.is_some() && self.biases_gradient.is_some() {
            Some(vec![
                (&mut self.weights, self.weights_gradient.as_ref().unwrap()),
                (&mut self.biases, self.biases_gradient.as_ref().unwrap()),
            ])
        } else {
            None
        }
    }
}
