//! Loss (cost) functions for neural network training.
//!
//! This module provides the `LossFunction` trait and implementations for common loss functions
//! used in supervised learning. Each loss function measures the discrepancy between predicted
//! outputs and true labels, with their derivatives used for backpropagation during training.
//!
//! # Available Loss Functions
//!
//! - **MSE** (`MeanSquaredError`): L2 loss - penalizes large errors quadratically
//! - **MAE** (`MeanAbsoluteError`): L1 loss - robust to outliers, linear penalty
//! - **BCE** (`BinaryCrossEntropy`): For binary classification with sigmoid output
//! - **CCE** (`CategoricalCrossEntropy`): For multi-class classification with softmax output
//! - **Huber Loss** (`HuberLoss`): Hybrid MSE/MAE - quadratic for small errors, linear for large
//! - **Hinge Loss** (`HingeLoss`): For SVM-style margin-based classification
//!
//! # Performance
//!
//! All loss functions leverage Rayon's parallel iterators for batch processing, automatically
//! distributing computation across available CPU cores when processing multiple samples.
//!
//! # Numerical Stability
//!
//! Cross-entropy losses (BCE, CCE) use epsilon clamping (`1e-7`) to prevent `log(0)` and
//! division by zero errors during gradient computation.

use crate::errors::MathError;
use crate::tensor::Tensor;
use ndarray::parallel::prelude::*;
use ndarray::{Axis, Zip};

/// Trait defining the interface for loss functions.
///
/// All loss functions must implement this trait to be used for training neural networks.
/// Both methods operate on tensors representing batches of samples.
///
/// # Parameters
///
/// - `t`: True labels (ground truth)
/// - `z`: Predicted outputs from the model
///
/// # Methods
///
/// - `function`: Computes the scalar loss value for a batch
/// - `derivative`: Computes the gradient w.r.t. predictions for backpropagation
///
/// # Thread Safety
///
/// The `Send + Sync` bounds allow loss functions to be safely shared across threads,
/// enabling parallel batch processing during training.
///
/// # Shape Requirements
///
/// Both `t` and `z` must have identical shapes, or an error is returned. The first
/// dimension is typically the batch dimension.
pub trait LossFunction: Send + Sync {
    /// Computes the loss value for a batch of samples.
    ///
    /// # Parameters
    ///
    /// - `t`: True labels tensor (shape: `[batch_size, ...]`)
    /// - `z`: Predicted outputs tensor (shape: `[batch_size, ...]`)
    ///
    /// # Returns
    ///
    /// A scalar f32 representing the average loss across the batch, or an error
    /// if shapes don't match or computation fails.
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError>;

    /// Computes the gradient of the loss w.r.t. predictions.
    ///
    /// # Parameters
    ///
    /// - `t`: True labels tensor
    /// - `z`: Predicted outputs tensor
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input, containing the gradient values
    /// for backpropagation, or an error if shapes don't match.
    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Mean Squared Error (MSE) loss function.
///
/// # Mathematical Definition
///
/// - **Function**: `L(t,z) = (1/n) * Σ(t_i - z_i)²`
/// - **Derivative**: `∂L/∂z_i = -2(t_i - z_i) / n`
///
/// # Properties
///
/// - L2 loss - penalizes large errors quadratically
/// - Sensitive to outliers due to squaring
/// - Commonly used for regression tasks
/// - Differentiable everywhere
///
/// # Use Cases
///
/// - Regression problems
/// - When large errors should be heavily penalized
/// - Gaussian noise assumptions
pub struct MeanSquaredError;

impl LossFunction for MeanSquaredError {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape {
            return Err(MathError::CostFunctionBatchShapeMismatch);
        }

        // Parallel computation across batch dimension
        let sum: f32 = t
            .data
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(z.data.axis_iter(Axis(0)).into_par_iter())
            .map(|(t_i, z_i)| -> Result<f32, MathError> {
                let t_i = Tensor {
                    dimension: t_i.ndim(),
                    shape: t_i.shape().to_vec(),
                    data: t_i.to_owned(),
                };
                let z_i = Tensor {
                    dimension: z_i.ndim(),
                    shape: z_i.shape().to_vec(),
                    data: z_i.to_owned(),
                };
                // Σ(t_i - z_i)²
                let diff = (&t_i - &z_i).map_err(|_| MathError::TensorSubtractionFailed)?;
                Ok(diff.data.mapv(|x| x.powi(2)).sum())
            })
            .collect::<Result<Vec<f32>, MathError>>()?
            .into_iter()
            .sum();

        // (1/n) * Σ(...)
        Ok(sum / (t.length() as f32))
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError> {
        if t.shape != z.shape {
            return Err(MathError::DerivativeShapeMismatch);
        }
        let n = t.length() as f32;
        let diff = (t - z).map_err(|_| MathError::TensorSubtractionFailed)?;
        // ∂L/∂z_i = -2(t_i - z_i) / n
        let gradients = diff.data.mapv(|x| -x * 2.0 / n);
        Ok(Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        })
    }
}

/// Mean Absolute Error (MAE) loss function.
///
/// # Mathematical Definition
///
/// - **Function**: `L(t,z) = (1/n) * Σ|t_i - z_i|`
/// - **Derivative**: `∂L/∂z_i = -sign(t_i - z_i) / n`
///
/// # Properties
///
/// - L1 loss - penalizes errors linearly
/// - More robust to outliers than MSE
/// - Not differentiable at zero (uses sign function)
/// - Gradients have constant magnitude regardless of error size
///
/// # Use Cases
///
/// - Regression with outliers in data
/// - When all errors should be weighted equally
/// - Robust optimization scenarios
pub struct MeanAbsoluteError;

impl LossFunction for MeanAbsoluteError {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape {
            return Err(MathError::CostFunctionBatchShapeMismatch);
        }

        // Parallel computation across batch dimension
        let sum: f32 = t
            .data
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(z.data.axis_iter(Axis(0)).into_par_iter())
            .map(|(t_i, z_i)| -> Result<f32, MathError> {
                let t_i = Tensor {
                    dimension: t_i.ndim(),
                    shape: t_i.shape().to_vec(),
                    data: t_i.to_owned(),
                };
                let z_i = Tensor {
                    dimension: z_i.ndim(),
                    shape: z_i.shape().to_vec(),
                    data: z_i.to_owned(),
                };
                // Σ|t_i - z_i|
                let diff = (&t_i - &z_i).map_err(|_| MathError::TensorSubtractionFailed)?;
                Ok(diff.data.mapv(|x| x.abs()).sum())
            })
            .collect::<Result<Vec<f32>, MathError>>()?
            .into_iter()
            .sum();

        // (1/n) * Σ|...|
        Ok(sum / (t.length() as f32))
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError> {
        if t.shape != z.shape {
            return Err(MathError::DerivativeShapeMismatch);
        }
        let diff = (t - z).map_err(|_| MathError::TensorSubtractionFailed)?;
        // ∂L/∂z_i = -sign(t_i - z_i) / n
        // At zero: gradient is undefined, we return 0
        let gradients = diff.data.mapv(|x| {
            if x == 0.0 {
                0.0
            } else {
                -(x.abs() / x) / t.length() as f32
            }
        });
        Ok(Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        })
    }
}

/// Binary Cross-Entropy (BCE) loss function.
///
/// # Mathematical Definition
///
/// - **Function**: `L(t,z) = -(1/n) * Σ[t_i·log(z_i) + (1-t_i)·log(1-z_i)]`
/// - **Derivative**: `∂L/∂z_i = -[t_i/z_i - (1-t_i)/(1-z_i)] / n`
///
/// # Properties
///
/// - Designed for binary classification tasks
/// - Expects labels in {0, 1} and predictions in (0, 1)
/// - Typically used with sigmoid activation in output layer
/// - Numerically stabilized with epsilon clamping to prevent `log(0)`
///
/// # Numerical Stability
///
/// Predictions are clamped to `[1e-7, 1-1e-7]` to avoid `log(0)` and division by zero.
///
/// # Use Cases
///
/// - Binary classification problems
/// - Multi-label classification (independent binary decisions per label)
/// - When output layer uses sigmoid activation
pub struct BinaryCrossEntropy;

impl LossFunction for BinaryCrossEntropy {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape {
            return Err(MathError::CostFunctionBatchShapeMismatch);
        }
        // Clamp predictions to prevent log(0)
        let epsilon = 1e-7;

        // Parallel computation across batch dimension
        let sum: f32 = t
            .data
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(z.data.axis_iter(Axis(0)).into_par_iter())
            .map(|(t_i, z_i)| -> Result<f32, MathError> {
                // L = -Σ[t·log(z) + (1-t)·log(1-z)]
                let sum: f32 = t_i
                    .iter()
                    .zip(z_i.iter())
                    .map(|(t_j, z_j)| {
                        let z_clamped = z_j.clamp(epsilon, 1.0 - epsilon);
                        t_j * z_clamped.ln() + (1.0 - t_j) * (1.0 - z_clamped).ln()
                    })
                    .sum();
                Ok(-sum)
            })
            .collect::<Result<Vec<f32>, MathError>>()?
            .into_iter()
            .sum();

        // (1/n) * Σ(...)
        Ok(sum / (t.length() as f32))
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError> {
        if t.shape != z.shape {
            return Err(MathError::DerivativeShapeMismatch);
        }

        let epsilon = 1e-7;

        // ∂L/∂z = -[t/z - (1-t)/(1-z)] / n
        let gradients = Zip::from(&t.data).and(&z.data).par_map_collect(|t_i, z_i| {
            let z_clamped = z_i.clamp(epsilon, 1.0 - epsilon);
            -((t_i / z_clamped) - ((1.0 - t_i) / (1.0 - z_clamped))) / t.length() as f32
        });

        Ok(Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        })
    }
}

/// Categorical Cross-Entropy (CCE) loss function.
///
/// # Mathematical Definition
///
/// - **Function**: `L(t,z) = -(1/n) * ΣΣ t_i,k·log(z_i,k)`
/// - **Derivative**: `∂L/∂z_i,k = -t_i,k / z_i,k / n`
///
/// # Properties
///
/// - Designed for multi-class classification (mutually exclusive classes)
/// - Expects one-hot encoded labels: `t_i,k ∈ {0,1}` with `Σt_i,k = 1`
/// - Expects softmax-normalized predictions: `z_i,k ∈ (0,1)` with `Σz_i,k = 1`
/// - Typically used with softmax activation in output layer
/// - Numerically stabilized with epsilon clamping
///
/// # Numerical Stability
///
/// Predictions are clamped to `[1e-7, 1-1e-7]` to avoid `log(0)`.
///
/// # Use Cases
///
/// - Multi-class classification (single label per sample)
/// - When output layer uses softmax activation
/// - Image classification, NLP classification tasks
pub struct CategoricalCrossEntropy;

impl LossFunction for CategoricalCrossEntropy {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape {
            return Err(MathError::CostFunctionBatchShapeMismatch);
        }
        // Clamp predictions to prevent log(0)
        let epsilon = 1e-7;

        // Parallel computation across batch dimension
        let sum: f32 = t
            .data
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(z.data.axis_iter(Axis(0)).into_par_iter())
            .map(|(t_i, z_i)| -> Result<f32, MathError> {
                // L = -Σ t_k·log(z_k)
                let sum: f32 = t_i
                    .iter()
                    .zip(z_i.iter())
                    .map(|(t_j, z_j)| {
                        let z_clamped = z_j.clamp(epsilon, 1.0 - epsilon);
                        t_j * z_clamped.ln()
                    })
                    .sum();
                Ok(-sum)
            })
            .collect::<Result<Vec<f32>, MathError>>()?
            .into_iter()
            .sum();

        // (1/n) * Σ(...)
        Ok(sum / (t.length() as f32))
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError> {
        if t.shape != z.shape {
            return Err(MathError::DerivativeShapeMismatch);
        }

        let epsilon = 1e-7;

        // ∂L/∂z = -t / z / n
        let gradients = Zip::from(&t.data).and(&z.data).par_map_collect(|t_i, z_i| {
            let z_clamped = z_i.clamp(epsilon, 1.0 - epsilon);
            -(t_i / z_clamped) / t.length() as f32
        });

        Ok(Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        })
    }
}

/// Huber Loss function.
///
/// # Mathematical Definition
///
/// ```text
/// L(t,z) = (1/n) * Σ L_δ(t_i - z_i)
///
/// where L_δ(x) = {
///   0.5·x²          if |x| ≤ δ
///   δ·|x| - 0.5·δ²  if |x| > δ
/// }
/// ```
///
/// Derivative:
/// ```text
/// ∂L/∂z_i = {
///   (z_i - t_i) / n           if |t_i - z_i| ≤ δ
///   -δ·sign(t_i - z_i) / n    if |t_i - z_i| > δ
/// }
/// ```
///
/// # Properties
///
/// - Hybrid loss combining MSE and MAE characteristics
/// - Quadratic for small errors (|x| ≤ δ): smooth, differentiable
/// - Linear for large errors (|x| > δ): robust to outliers
/// - Transition point controlled by delta parameter (δ)
/// - More robust than MSE, smoother gradient than MAE
///
/// # Parameters
///
/// - `delta`: Threshold determining quadratic/linear transition point
///
/// # Use Cases
///
/// - Regression with occasional outliers
/// - When you want MSE smoothness for typical errors but MAE robustness for outliers
/// - Object detection bounding box regression
pub struct HuberLoss {
    /// Threshold δ for switching between quadratic and linear loss.
    pub delta: f32,
}

impl LossFunction for HuberLoss {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape {
            return Err(MathError::CostFunctionBatchShapeMismatch);
        }

        // Parallel computation across batch dimension
        let sum: f32 = t
            .data
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(z.data.axis_iter(Axis(0)).into_par_iter())
            .map(|(t_i, z_i)| -> Result<f32, MathError> {
                let sum: f32 = t_i
                    .iter()
                    .zip(z_i.iter())
                    .map(|(t_j, z_j)| {
                        let diff = t_j - z_j;
                        // L_δ(x) = 0.5·x² if |x| ≤ δ, else δ·|x| - 0.5·δ²
                        if diff.abs() <= self.delta {
                            diff.powi(2) / 2.0
                        } else {
                            self.delta * diff.abs() - self.delta.powi(2) / 2.0
                        }
                    })
                    .sum();
                Ok(sum)
            })
            .collect::<Result<Vec<f32>, MathError>>()?
            .into_iter()
            .sum();

        // (1/n) * Σ(...)
        Ok(sum / (t.length() as f32))
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError> {
        if t.shape != z.shape {
            return Err(MathError::DerivativeShapeMismatch);
        }

        // ∂L/∂z = (z - t)/n if |t - z| ≤ δ, else -δ·sign(t - z)/n
        let gradients = Zip::from(&t.data).and(&z.data).par_map_collect(|t_i, z_i| {
            let diff = t_i - z_i;
            if diff.abs() <= self.delta {
                (z_i - t_i) / t.length() as f32
            } else {
                (-self.delta * diff.abs() / diff) / t.length() as f32
            }
        });

        Ok(Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        })
    }
}

/// Hinge Loss function.
///
/// # Mathematical Definition
///
/// - **Function**: `L(t,z) = (1/n) * Σ max(0, 1 - t_i·z_i)`
/// - **Derivative**: `∂L/∂z_i = -t_i / n` if `t_i·z_i < 1`, else `0`
///
/// # Properties
///
/// - Designed for binary classification with margin-based learning
/// - Expects labels in {-1, +1} (not {0, 1})
/// - Penalizes predictions that are on the wrong side or too close to the decision boundary
/// - Zero loss when correct prediction with sufficient margin (`t·z ≥ 1`)
/// - Used in Support Vector Machines (SVMs)
///
/// # Margin Interpretation
///
/// - `t·z > 1`: Correct with margin → loss = 0
/// - `0 < t·z < 1`: Correct but insufficient margin → linear penalty
/// - `t·z < 0`: Incorrect prediction → larger linear penalty
///
/// # Use Cases
///
/// - Binary classification with margin enforcement
/// - Support Vector Machines (SVMs)
/// - When you want to encourage confident predictions
pub struct HingeLoss;

impl LossFunction for HingeLoss {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape {
            return Err(MathError::CostFunctionBatchShapeMismatch);
        }

        // Parallel computation across batch dimension
        let sum: f32 = t
            .data
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(z.data.axis_iter(Axis(0)).into_par_iter())
            .map(|(t_i, z_i)| -> Result<f32, MathError> {
                // L = Σ max(0, 1 - t·z)
                let sum: f32 = t_i
                    .iter()
                    .zip(z_i.iter())
                    .map(|(t_j, z_j)| 0.0_f32.max(1.0 - t_j * z_j))
                    .sum();
                Ok(sum)
            })
            .collect::<Result<Vec<f32>, MathError>>()?
            .into_iter()
            .sum();

        // (1/n) * Σ(...)
        Ok(sum / (t.length() as f32))
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError> {
        if t.shape != z.shape {
            return Err(MathError::DerivativeShapeMismatch);
        }

        // ∂L/∂z = -t/n if t·z < 1, else 0
        let gradients = Zip::from(&t.data).and(&z.data).par_map_collect(|t_i, z_i| {
            let x = if t_i * z_i < 1.0 { -t_i } else { 0.0 };
            x / t.length() as f32
        });

        Ok(Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        })
    }
}
