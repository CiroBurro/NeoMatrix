//! Weight initialization strategies for neural network layers.
//!
//! Proper weight initialization is critical for training deep neural networks. This module
//! provides three initialization strategies, each optimized for different activation functions
//! and network architectures.
//!
//! # Available Strategies
//!
//! - **[`Init::Random`]**: Simple uniform random initialization (legacy, not recommended)
//! - **[`Init::Xavier`]**: Xavier/Glorot initialization (optimal for Sigmoid/Tanh activations)
//! - **[`Init::He`]**: He initialization (optimal for ReLU/Leaky ReLU activations)
//!
//! # Why Initialization Matters
//!
//! Poor initialization can cause:
//! - **Vanishing gradients**: Gradients become too small, preventing learning in deep networks
//! - **Exploding gradients**: Gradients become too large, causing numerical instability
//! - **Dead neurons**: ReLU units that never activate, becoming permanently inactive
//!
//! Xavier and He initialization mitigate these issues by scaling weights based on layer dimensions.
//!
//! # Choosing an Initialization Strategy
//!
//! | Activation Function | Recommended Init | Reasoning |
//! |---------------------|------------------|-----------|
//! | ReLU, Leaky ReLU    | `Init::He`       | Accounts for ReLU's non-linearity (kills negative values) |
//! | Sigmoid, Tanh       | `Init::Xavier`   | Maintains variance for symmetric activations |
//! | Linear (none)       | `Init::Xavier`   | Balanced variance preservation |
//!
//! # Example
//!
//! ```rust,ignore
//! use neomatrix_core::layers::{dense::Dense, init::Init};
//!
//! // He initialization for a ReLU network
//! let layer1 = Dense::new(784, 256, Some(Init::He), None);
//!
//! // Xavier initialization for a Sigmoid network
//! let layer2 = Dense::new(256, 128, Some(Init::Xavier), None);
//!
//! // Custom random initialization (not recommended for deep networks)
//! let layer3 = Dense::new(128, 10, Some(Init::Random), Some(-0.1..0.1));
//! ```

use crate::tensor::Tensor;
use ndarray::Array2;
use ndarray_rand::rand_distr::{Distribution, Normal};
use std::ops::Range;

/// Weight initialization strategies for neural network layers.
///
/// Each variant implements a different initialization algorithm, optimized for specific
/// activation functions and network architectures.
#[derive(Clone, Debug)]
pub enum Init {
    /// Simple uniform random initialization.
    ///
    /// Samples weights uniformly from a user-specified range (default: -1 to 1).
    /// **Not recommended** for deep networks due to potential gradient issues.
    Random,

    /// Xavier (Glorot) initialization.
    ///
    /// Optimal for **Sigmoid** and **Tanh** activation functions. Weights are sampled from
    /// a normal distribution with zero mean and variance:
    ///
    /// `Var(W) = 2 / (n_in + n_out)`
    ///
    /// where `n_in` is the number of input features and `n_out` is the number of output features.
    ///
    /// # Mathematical Formula
    ///
    /// `W ~ N(0, sqrt(2 / (n_in + n_out)))`
    ///
    /// This maintains roughly equal variance for activations and gradients across layers.
    Xavier,

    /// He initialization.
    ///
    /// Optimal for **ReLU** and **Leaky ReLU** activation functions. Weights are sampled from
    /// a normal distribution with zero mean and variance:
    ///
    /// `Var(W) = 2 / n_in`
    ///
    /// The increased variance (compared to Xavier) compensates for ReLU's property of zeroing
    /// out negative values, which would otherwise reduce signal variance by ~50%.
    ///
    /// # Mathematical Formula
    ///
    /// `W ~ N(0, sqrt(2 / n_in))`
    ///
    /// This keeps the signal variance roughly constant through layers with ReLU activations.
    He,
}

impl Init {
    /// Initializes a weight tensor using the selected strategy.
    ///
    /// # Parameters
    ///
    /// - `in_feat`: Number of input features (rows in weight matrix)
    /// - `out_feat`: Number of output features (columns in weight matrix)
    /// - `rg`: Optional range for `Init::Random`. Ignored for `Xavier` and `He`.
    ///   Defaults to `-1.0..1.0` if `None` when using `Random`.
    ///
    /// # Returns
    ///
    /// A 2D tensor with shape `[in_feat, out_feat]` containing initialized weights.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use neomatrix_core::layers::init::Init;
    ///
    /// // He initialization for a 128 → 64 layer
    /// let weights = Init::He.init(128, 64, None);
    /// assert_eq!(weights.shape(), &[128, 64]);
    /// ```
    pub fn init(&self, in_feat: usize, out_feat: usize, rg: Option<Range<f32>>) -> Tensor {
        match self {
            Init::Random => Tensor::random(vec![in_feat, out_feat], rg.unwrap_or(-1.0..1.0)),
            Init::Xavier => {
                // Xavier initialization: W ~ N(0, sqrt(2 / (n_in + n_out)))
                let std_dev = (2.0 / (in_feat + out_feat) as f32).sqrt();
                let dist = Normal::new(0.0f32, std_dev).unwrap();
                let mut rng = rand::rng();

                let data = Array2::from_shape_fn((in_feat, out_feat), |_| dist.sample(&mut rng))
                    .into_dyn();
                Tensor {
                    dimension: 2,
                    shape: vec![in_feat, out_feat],
                    data,
                }
            }
            Init::He => {
                // He initialization: W ~ N(0, sqrt(2 / n_in))
                let std_dev = (2.0 / in_feat as f32).sqrt();
                let dist = Normal::new(0.0f32, std_dev).unwrap();
                let mut rng = rand::rng();

                let data = Array2::from_shape_fn((in_feat, out_feat), |_| dist.sample(&mut rng))
                    .into_dyn();
                Tensor {
                    dimension: 2,
                    shape: vec![in_feat, out_feat],
                    data,
                }
            }
        }
    }
}
