/// This module defines cost functions for neural networks.
/// It implements various cost functions like Mean Squared Error, Mean Absolute Error, 
/// Binary Cross-Entropy, Huber Loss and Hinge Loss, providing both regular and parallel computation methods.
/// Necessary imports
use crate::structures::tensor::Tensor;
use ndarray::parallel::prelude::*;
use ndarray::{ArrayD, Axis};
use pyo3::prelude::*;

/// Trait defining the interface for cost functions
/// 
/// # Methods:
/// * `function` - Regular computation for single sample
/// * `function_batch` - Regular computation for batch of samples
/// * `par_function_batch` - Parallel computation for batch of samples
/// * `derivative` - Derivative computation for backpropagation
pub trait CostFunction: Send + Sync {
    fn function(&self, t: &Tensor, z: &Tensor) -> f64;
    fn function_batch(&self, t: &Tensor, z: &Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 2 {
            panic!("Tensors shape have to be the same and dimension 2 for batch computation of the cost function")
        }

        let (m, _) = (t.shape[0], t.shape[1]);
        let sum: f64 = t
            .data
            .axis_iter(Axis(0))
            .zip(z.data.axis_iter(Axis(0)))
            .map(|(t_i, z_i)| {
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
            .sum();
        sum / m as f64
    }
    fn par_function_batch(&self, t: &Tensor, z: &Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 2 {
            panic!("Tensors shape have to be the same and dimension 2 for batch computation of the cost function")
        }

        let (m, _) = (t.shape[0], t.shape[1]);
        let sum: f64 = t
            .data
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(z.data.axis_iter(Axis(0)).into_par_iter())
            .map(|(t_i, z_i)| {
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
            .sum();
        sum / m as f64
    }
    fn derivative(&self, t: &Tensor, z: &Tensor) -> Tensor;
}

/// Cost enum
///
/// This python-accessible enum is used for cost function selection
///
/// # Variants
/// * `MeanSquaredError()`
/// * `MeanAbsoluteError()`
/// * `BinaryCrossEntropy()`
/// * `CategoricalCrossEntropy()`
/// * `HuberLoss()`
/// * `HingeLoss()`
#[pyclass]
#[derive(Clone, Debug)]
pub enum Cost {
    MeanSquaredError(),
    MeanAbsoluteError(),
    BinaryCrossEntropy(),
    CategoricalCrossEntropy(),
    HuberLoss {delta: f64},
    HingeLoss(),
}

/// Function to get and compute cost between two tensors
/// 
/// # Arguments
/// * `cost` - Type of cost function to use
/// * `t` - Target tensor
/// * `z` - Predicted tensor
/// * `parallel` - Option to use parallel computation
/// * `batch` - Option to use batch computation
///
/// # Returns
/// * `f64` - Error of the model
///
/// # Python usage:
///     ```python
///     from neomatrix import Tensor, Cost, get_cost
///     t = Tensor([4], [1, 2, 3, 4])
///     z = Tensor([4], [1.1, 2.1, 2.9, 4.2])
///     cost = get_cost(Cost.MeanSquaredError, t, z, parallel=True, batch=True)
///     ```
#[pyfunction]
pub fn get_cost(
    cost: Cost,
    t: &Tensor,
    z: &Tensor,
    parallel: Option<bool>,
    batch_processing: Option<bool>,
) -> f64 {
    let parallel = parallel.unwrap_or(false);
    let batch_processing = batch_processing.unwrap_or(true);
    let f: Box<dyn CostFunction> = match cost {
        Cost::MeanSquaredError() => Box::new(MeanSquaredError),
        Cost::MeanAbsoluteError() => Box::new(MeanAbsoluteError),
        Cost::BinaryCrossEntropy() => Box::new(BinaryCrossEntropy),
        Cost::CategoricalCrossEntropy() => Box::new(CategoricalCrossEntropy),
        Cost::HuberLoss { delta } => Box::new(HuberLoss { delta }),
        Cost::HingeLoss() => Box::new(HingeLoss),
    };
    
    if !batch_processing {
        f.function(t, z)
    } else if parallel {
        f.par_function_batch(t, z)
    } else {
        f.function_batch(t, z)
    }
}

/// Mean Squared Error cost function
/// f(t,z) = (1/n) * Σ(t_i - z_i)^2
pub struct MeanSquaredError;
impl CostFunction for MeanSquaredError {
    fn function(&self, t: &Tensor, z: &Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 1 {
            panic!("Tensors shape have to be the same and dimension 1 for computation of the cost function")
        }
        let nomin = (t-z)
            .expect("Tensors subtraction failed")
            .data
            .mapv(|x| x.powi(2))
            .sum();
        let denom = t.shape[0] as f64;
        nomin / denom
    }
    
    fn derivative(&self, t: &Tensor, z: &Tensor) -> Tensor {
        if t.shape != z.shape {
            panic!("Tensors shape have to be the same for computation of the derivative of the cost function")
        }

        let n = check_dimension(t);
        
        let gradients = (t-z)
            .expect("Tensors subtraction failed")
            .data
            .mapv(|x| -x * 2.0 / n);
        Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        }
    }
}


/// Mean Absolute Error cost function
/// `f(t,z) = (1/n) * Σ|t_i - z_i|`
pub struct MeanAbsoluteError;
impl CostFunction for MeanAbsoluteError {
    fn function(&self, t: &Tensor, z: &Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 1 {
            panic!("Tensors shape have to be the same and dimension 1 for computation of the cost function")
        }
        let nomin = (t-z)
            .expect("Tensors subtraction failed")
            .data
            .mapv(|x| x.abs())
            .sum();
        let denom = t.shape[0] as f64;
        nomin / denom
    }
    
    fn derivative(&self, t: &Tensor, z: &Tensor) -> Tensor {
        if t.shape != z.shape {
            panic!("Tensors shape have to be the same for computation of the derivative of the cost function")
        }

        let n = check_dimension(t);
        
        let gradients = (t-z)
            .expect("Tensors subtraction failed")
            .data
            .mapv(|x| -(x.abs() / x) / n);
        Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        }
    }
}

/// Binary Cross-Entropy cost function
/// `f(t,z) = -(1/n) * Σ(t_i * log(z_i) + (1-t_i) * log(1-z_i))`
pub struct BinaryCrossEntropy;
impl CostFunction for BinaryCrossEntropy {
    fn function(&self, t: &Tensor, z: &Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 1 {
            panic!("Tensors shape have to be the same and dimension 1 for computation of the cost function")
        }

        let sum: f64 = t
            .data
            .to_owned()
            .iter()
            .zip(z.data.to_owned().iter())
            .map(|(t_i, z_i)| t_i * z_i.ln() + (1.0 - t_i) * (1.0 - z_i).ln())
            .sum::<f64>();

        -sum / t.shape[0] as f64
    }
    
    fn derivative(&self, t: &Tensor, z: &Tensor) -> Tensor {
        if t.shape != z.shape {
            panic!("Tensors shape have to be the same for computation of the derivative of the cost function")
        }

        let n = check_dimension(t);

        let gradients_vec = t.data.iter().zip(z.data.iter()).map(|(t_i, z_i)| {
            -((t_i / z_i) - ((1.0 - t_i) / (1.0 - z_i))) / n
        }).collect::<Vec<f64>>();

        let gradients = ArrayD::from_shape_vec(t.shape.clone(), gradients_vec).unwrap();

        Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        }
    }
}

/// Categorical Cross Entropy cost function
/// `f(t, z) = -(1/n) * Σ(Σ(t_ik * log(z_ik)))`
pub struct CategoricalCrossEntropy;
impl CostFunction for CategoricalCrossEntropy {
    fn function(&self, t: &Tensor, z: &Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 1 {
            panic!("Tensors shape have to be the same and dimension 1 for computation of the cost function")
        }
        let sum: f64 = t
            .data
            .to_owned()
            .iter()
            .zip(z.data.to_owned().iter())
            .map(|(t_k, z_k)| t_k * z_k.ln())
            .sum::<f64>();

        -sum
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Tensor {
        if t.shape != z.shape {
            panic!("Tensors shape have to be the same for computation of the derivative of the cost function")
        }

        let n = check_dimension(t);

        let gradients_vec = t.data.iter().zip(z.data.iter()).map(|(t_i, z_i)| {
            -(t_i / z_i) / n
        }).collect::<Vec<f64>>();

        let gradients = ArrayD::from_shape_vec(t.shape.clone(), gradients_vec).unwrap();

        Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        }
    }
}

/// Huber Loss cost function
/// ```
/// f(t,z) = (1/n) * Σ L_δ(t_i - z_i)
/// where L_δ(x) = 0.5 * x^2 if |x| ≤ δ
///                 δ|x| - 0.5δ^2 if |x| > δ
/// ```
pub struct HuberLoss { pub delta: f64 }
impl CostFunction for HuberLoss {
    fn function(&self, t: &Tensor, z: &Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 1 {
            panic!("Tensors shape have to be the same and dimension 1 for computation of the cost function")
        }


        let sum: f64 = t
            .data
            .iter()
            .zip(z.data.iter())
            .map(|(t_i, z_i)| {
                if (t_i - z_i).abs() <= self.delta {
                    (t_i - z_i).powf(2.0) / 2.0
                } else {
                    self.delta * (t_i - z_i).abs() - (self.delta.powf(2.0) / 2.0)
                }
            })
            .sum();

        sum / t.shape[0] as f64
    }

    fn derivative(&self, t: &Tensor, z: &Tensor) -> Tensor {
        if t.shape != z.shape {
            panic!("Tensors shape have to be the same and dimension 1 for computation of the derivative of the cost function")
        }
        
        let n = check_dimension(t);

        let gradients_vec = t.data.iter().zip(z.data.iter()).map(|(t_i, z_i)| {
            if (t_i - z_i).abs() <= self.delta {
                (z_i - t_i) / n
            } else {
                if (t_i - z_i) == 0.0 {
                    return 0.0;
                }
                (- self.delta * (t_i - z_i).abs() / (t_i - z_i)) / n
            }
        }).collect::<Vec<f64>>();

        let gradients = ArrayD::from_shape_vec(t.shape.clone(), gradients_vec).unwrap();

        Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        }
    }
}

/// Hinge Loss cost function
/// `f(t,z) = (1/n) * Σ max(0, 1 - t_i * z_i)`
pub struct HingeLoss;
impl CostFunction for HingeLoss {
    fn function(&self, t: &Tensor, z: &Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 1 {
            panic!("Tensors shape have to be the same and dimension 1 for computation of the cost function")
        }

        let sum: f64 = t
            .data
            .iter()
            .zip(z.data.iter())
            .map(|(t_i, z_i)| 0.0_f64.max(1.0 - (t_i * z_i)))
            .sum();

        sum / t.shape[0] as f64
    }
    
    fn derivative(&self, t: &Tensor, z: &Tensor) -> Tensor {
        if t.shape != z.shape {
            panic!("Tensors shape have to be the same and dimension 1 for computation of the derivative of the cost function")
        }

        let n = check_dimension(t);

        let gradients_vec = t.data.iter().zip(z.data.iter()).map(|(t_i, z_i)| {
            let x = if t_i * z_i < 1.0 {
                -t_i
            } else {
                0.0
            };

            x / n
        }).collect::<Vec<f64>>();

        let gradients = ArrayD::from_shape_vec(t.shape.clone(), gradients_vec).unwrap();

        Tensor {
            dimension: gradients.ndim(),
            shape: gradients.shape().to_vec(),
            data: gradients,
        }
    }
}


/// Function to check dimensions of a tensor
///   ! Not a Python function !
///
/// # Arguments
/// * `t` - Tensor to check
///
/// # Returns
/// * `f64` - Length of the tensor
fn check_dimension(t: &Tensor) -> f64 {
    
    let mut n = 1.0;
    
    if t.dimension == 1 {
        n = t.shape[0] as f64
    } else if t.dimension == 2 {
        n = (t.shape[0] * t.shape[1]) as f64
    }
    
    n
}
