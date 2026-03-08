/// Cost functions for neural networks.
/// Implements: MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy, Huber Loss, Hinge Loss.
use crate::errors::MathError;
use crate::tensor::Tensor;
use ndarray::parallel::prelude::*;
use ndarray::{ArrayD, Axis};

/// Trait defining the interface for cost functions
///
/// # Methods:
/// * `function` - Computation for a single 1D sample
/// * `function_batch` - Parallel computation over a 2D batch (default impl)
/// * `derivative` - Gradient w.r.t. predictions for backpropagation
pub trait CostFunction: Send + Sync {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError>;

    fn function_batch(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape || t.dimension != 2 {
            return Err(MathError::CostFunctionBatchShapeMismatch);
        }
        let m = t.shape[0];
        let sum: f32 = t
            .data
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(z.data.axis_iter(Axis(0)).into_par_iter())
            .map(|(t_i, z_i)| -> Result<f32, MathError> {
                let t_i = Tensor {
                    dimension: 1,
                    shape: vec![t.shape[1]],
                    data: t_i.to_owned(),
                };
                let z_i = Tensor {
                    dimension: 1,
                    shape: vec![z.shape[1]],
                    data: z_i.to_owned(),
                };
                self.function(&t_i, &z_i)
            })
            .collect::<Result<Vec<f32>, MathError>>()?
            .into_iter()
            .sum();
        Ok(sum / m as f32)
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError>;
}

/// Enum for cost function selection
#[derive(Clone, Debug)]
pub enum Cost {
    MeanSquaredError,
    MeanAbsoluteError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    HuberLoss { delta: f32 },
    HingeLoss,
}

impl Cost {
    pub fn name(&self) -> &str {
        match self {
            Cost::MeanSquaredError => "MSE",
            Cost::MeanAbsoluteError => "MAE",
            Cost::BinaryCrossEntropy => "BCE",
            Cost::CategoricalCrossEntropy => "CCE",
            Cost::HuberLoss { .. } => "HuberLoss",
            Cost::HingeLoss => "HingeLoss",
        }
    }
}

impl TryFrom<&str> for Cost {
    type Error = MathError;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "MSE" => Ok(Cost::MeanSquaredError),
            "MAE" => Ok(Cost::MeanAbsoluteError),
            "BCE" => Ok(Cost::BinaryCrossEntropy),
            "CCE" => Ok(Cost::CategoricalCrossEntropy),
            "HingeLoss" => Ok(Cost::HingeLoss),
            other => Err(MathError::UnknownCostFunction(other.to_string())),
        }
    }
}

/// Helper: dispatch a `Cost` variant to a `Box<dyn CostFunction>`
fn cost_to_fn(cost: &Cost) -> Box<dyn CostFunction> {
    match cost {
        Cost::MeanSquaredError => Box::new(MeanSquaredError),
        Cost::MeanAbsoluteError => Box::new(MeanAbsoluteError),
        Cost::BinaryCrossEntropy => Box::new(BinaryCrossEntropy),
        Cost::CategoricalCrossEntropy => Box::new(CategoricalCrossEntropy),
        Cost::HuberLoss { delta } => Box::new(HuberLoss { delta: *delta }),
        Cost::HingeLoss => Box::new(HingeLoss),
    }
}

/// Compute the cost between two tensors
///
/// # Arguments
/// * `cost` - Cost function variant to use
/// * `t` - Target tensor
/// * `z` - Predicted tensor
/// * `batch_processing` - If `true` (default), uses the batch (parallel) method
pub fn get_cost(
    cost: &Cost,
    t: &Tensor,
    z: &Tensor,
    batch_processing: bool,
) -> Result<f32, MathError> {
    let f = cost_to_fn(cost);
    if batch_processing {
        f.function_batch(t, z)
    } else {
        f.function(t, z)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Mean Squared Error — `f(t,z) = (1/n) * Σ(t_i - z_i)²`
pub struct MeanSquaredError;
impl CostFunction for MeanSquaredError {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape || t.dimension != 1 {
            return Err(MathError::CostFunctionShapeMismatch);
        }
        let diff = (t - z).map_err(|_| MathError::TensorSubtractionFailed)?;
        let n = t.shape[0] as f32;
        Ok(diff.data.mapv(|x| x.powi(2)).sum() / n)
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError> {
        if t.shape != z.shape {
            return Err(MathError::DerivativeShapeMismatch);
        }
        let n = t.length() as f32;
        let diff = (t - z).map_err(|_| MathError::TensorSubtractionFailed)?;
        let gradients = diff.data.mapv(|x| -x * 2.0 / n);
        Ok(Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        })
    }
}

/// Mean Absolute Error — `f(t,z) = (1/n) * Σ|t_i - z_i|`
pub struct MeanAbsoluteError;
impl CostFunction for MeanAbsoluteError {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape || t.dimension != 1 {
            return Err(MathError::CostFunctionShapeMismatch);
        }
        let diff = (t - z).map_err(|_| MathError::TensorSubtractionFailed)?;
        let n = t.shape[0] as f32;
        Ok(diff.data.mapv(|x| x.abs()).sum() / n)
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError> {
        if t.shape != z.shape {
            return Err(MathError::DerivativeShapeMismatch);
        }
        let n = t.length() as f32;
        let diff = (t - z).map_err(|_| MathError::TensorSubtractionFailed)?;
        let gradients = diff.data.mapv(|x| -(x.abs() / x) / n);
        Ok(Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        })
    }
}

/// Binary Cross-Entropy — `f(t,z) = -(1/n) * Σ(t_i·log(z_i) + (1-t_i)·log(1-z_i))`
pub struct BinaryCrossEntropy;
impl CostFunction for BinaryCrossEntropy {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape || t.dimension != 1 {
            return Err(MathError::CostFunctionShapeMismatch);
        }
        let sum: f32 = t
            .data
            .iter()
            .zip(z.data.iter())
            .map(|(t_i, z_i)| t_i * z_i.ln() + (1.0 - t_i) * (1.0 - z_i).ln())
            .sum();
        Ok(-sum / t.shape[0] as f32)
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError> {
        if t.shape != z.shape {
            return Err(MathError::DerivativeShapeMismatch);
        }
        let n = t.length() as f32;
        let grad_vec: Vec<f32> = t
            .data
            .iter()
            .zip(z.data.iter())
            .map(|(t_i, z_i)| -((t_i / z_i) - ((1.0 - t_i) / (1.0 - z_i))) / n)
            .collect();
        let gradients = ArrayD::from_shape_vec(t.shape.clone(), grad_vec)
            .map_err(|_| MathError::GradientShapeMismatch)?;
        Ok(Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        })
    }
}

/// Categorical Cross-Entropy — `f(t,z) = -Σ(t_k·log(z_k))`
pub struct CategoricalCrossEntropy;
impl CostFunction for CategoricalCrossEntropy {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape || t.dimension != 1 {
            return Err(MathError::CostFunctionShapeMismatch);
        }
        let sum: f32 = t
            .data
            .iter()
            .zip(z.data.iter())
            .map(|(t_k, z_k)| t_k * z_k.ln())
            .sum();
        Ok(-sum)
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError> {
        if t.shape != z.shape {
            return Err(MathError::DerivativeShapeMismatch);
        }
        let n = t.length() as f32;
        let grad_vec: Vec<f32> = t
            .data
            .iter()
            .zip(z.data.iter())
            .map(|(t_i, z_i)| -(t_i / z_i) / n)
            .collect();
        let gradients = ArrayD::from_shape_vec(t.shape.clone(), grad_vec)
            .map_err(|_| MathError::GradientShapeMismatch)?;
        Ok(Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        })
    }
}

/// Huber Loss
/// ```
/// f(t,z) = (1/n) * Σ Lδ(t_i - z_i)
/// Lδ(x) = 0.5·x²        if |x| ≤ δ
///          δ·|x| - 0.5δ² if |x| > δ
/// ```
pub struct HuberLoss {
    pub delta: f32,
}
impl CostFunction for HuberLoss {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape || t.dimension != 1 {
            return Err(MathError::CostFunctionShapeMismatch);
        }
        let sum: f32 = t
            .data
            .iter()
            .zip(z.data.iter())
            .map(|(t_i, z_i)| {
                let diff = t_i - z_i;
                if diff.abs() <= self.delta {
                    diff.powi(2) / 2.0
                } else {
                    self.delta * diff.abs() - self.delta.powi(2) / 2.0
                }
            })
            .sum();
        Ok(sum / t.shape[0] as f32)
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError> {
        if t.shape != z.shape {
            return Err(MathError::DerivativeShapeMismatch);
        }
        let n = t.length() as f32;
        let grad_vec: Vec<f32> = t
            .data
            .iter()
            .zip(z.data.iter())
            .map(|(t_i, z_i)| {
                let diff = t_i - z_i;
                if diff.abs() <= self.delta {
                    (z_i - t_i) / n
                } else if diff == 0.0 {
                    0.0
                } else {
                    (-self.delta * diff.abs() / diff) / n
                }
            })
            .collect();
        let gradients = ArrayD::from_shape_vec(t.shape.clone(), grad_vec)
            .map_err(|_| MathError::GradientShapeMismatch)?;
        Ok(Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        })
    }
}

/// Hinge Loss — `f(t,z) = (1/n) * Σ max(0, 1 - t_i·z_i)`
pub struct HingeLoss;
impl CostFunction for HingeLoss {
    fn function(&self, t: &Tensor, z: &Tensor) -> Result<f32, MathError> {
        if t.shape != z.shape || t.dimension != 1 {
            return Err(MathError::CostFunctionShapeMismatch);
        }
        let sum: f32 = t
            .data
            .iter()
            .zip(z.data.iter())
            .map(|(t_i, z_i)| 0.0_f32.max(1.0 - t_i * z_i))
            .sum();
        Ok(sum / t.shape[0] as f32)
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Result<Tensor, MathError> {
        if t.shape != z.shape {
            return Err(MathError::DerivativeShapeMismatch);
        }
        let n = t.length() as f32;
        let grad_vec: Vec<f32> = t
            .data
            .iter()
            .zip(z.data.iter())
            .map(|(t_i, z_i)| {
                let x = if t_i * z_i < 1.0 { -t_i } else { 0.0 };
                x / n
            })
            .collect();
        let gradients = ArrayD::from_shape_vec(t.shape.clone(), grad_vec)
            .map_err(|_| MathError::GradientShapeMismatch)?;
        Ok(Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        })
    }
}
