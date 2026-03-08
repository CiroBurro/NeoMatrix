//! Test suite for loss functions
//!
//! Tests verify both forward pass (function) and backward pass (derivative)
//! for all loss functions: MSE, MAE, BCE, CCE, Huber, Hinge.

use crate::math::losses::{
    BinaryCrossEntropy, CategoricalCrossEntropy, HingeLoss, HuberLoss, LossFunction,
    MeanAbsoluteError, MeanSquaredError,
};
use crate::tensor::Tensor;

const EPSILON: f32 = 1e-5;

#[cfg(test)]
mod mse_tests {
    use super::*;

    #[test]
    fn mse_function_perfect_prediction() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let z = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        let mse = MeanSquaredError;
        let loss = mse.function(&t, &z).unwrap();

        assert_eq!(loss, 0.0);
    }

    #[test]
    fn mse_function_simple_case() {
        let t = Tensor::new(vec![2], vec![3.0, 5.0]).unwrap();
        let z = Tensor::new(vec![2], vec![2.0, 3.0]).unwrap();

        let mse = MeanSquaredError;
        let loss = mse.function(&t, &z).unwrap();

        assert!((loss - 2.5).abs() < EPSILON);
    }

    #[test]
    fn mse_function_2d() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let z = Tensor::new(vec![2, 2], vec![2.0, 3.0, 4.0, 5.0]).unwrap();

        let mse = MeanSquaredError;
        let loss = mse.function(&t, &z).unwrap();

        assert_eq!(loss, 1.0);
    }

    #[test]
    fn mse_function_batch() {
        let t = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let z = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let mse = MeanSquaredError;
        let loss = mse.function(&t, &z).unwrap();

        assert_eq!(loss, 0.0);
    }

    #[test]
    fn mse_function_negative_values() {
        let t = Tensor::new(vec![3], vec![-1.0, -2.0, -3.0]).unwrap();
        let z = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        let mse = MeanSquaredError;
        let loss = mse.function(&t, &z).unwrap();

        let expected = (4.0 + 16.0 + 36.0) / 3.0;
        assert!((loss - expected).abs() < EPSILON);
    }

    #[test]
    fn mse_function_shape_mismatch() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let z = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();

        let mse = MeanSquaredError;
        let result = mse.function(&t, &z);

        assert!(result.is_err());
    }

    #[test]
    fn mse_derivative_perfect_prediction() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let z = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        let mse = MeanSquaredError;
        let grad = mse.derivative(&t, &z).unwrap();

        assert_eq!(grad.data.as_slice().unwrap(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn mse_derivative_simple_case() {
        let t = Tensor::new(vec![2], vec![3.0, 5.0]).unwrap();
        let z = Tensor::new(vec![2], vec![2.0, 3.0]).unwrap();

        let mse = MeanSquaredError;
        let grad = mse.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2]);
        assert!((grad.data[0] - (-1.0)).abs() < EPSILON);
        assert!((grad.data[1] - (-2.0)).abs() < EPSILON);
    }

    #[test]
    fn mse_derivative_2d() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let z = Tensor::new(vec![2, 2], vec![2.0, 3.0, 4.0, 5.0]).unwrap();

        let mse = MeanSquaredError;
        let grad = mse.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2, 2]);
        for &val in grad.data.iter() {
            assert!((val - 0.5).abs() < EPSILON);
        }
    }

    #[test]
    fn mse_derivative_shape_mismatch() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let z = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();

        let mse = MeanSquaredError;
        let result = mse.derivative(&t, &z);

        assert!(result.is_err());
    }

    #[test]
    fn mse_derivative_preserves_shape() {
        let t = Tensor::new(vec![3, 4], vec![1.0; 12]).unwrap();
        let z = Tensor::new(vec![3, 4], vec![2.0; 12]).unwrap();

        let mse = MeanSquaredError;
        let grad = mse.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, t.shape);
        assert_eq!(grad.dimension, t.dimension);
    }
}

#[cfg(test)]
mod mae_tests {
    use super::*;

    #[test]
    fn mae_function_perfect_prediction() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let z = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        let mae = MeanAbsoluteError;
        let loss = mae.function(&t, &z).unwrap();

        assert_eq!(loss, 0.0);
    }

    #[test]
    fn mae_function_simple_case() {
        let t = Tensor::new(vec![2], vec![3.0, 5.0]).unwrap();
        let z = Tensor::new(vec![2], vec![2.0, 3.0]).unwrap();

        let mae = MeanAbsoluteError;
        let loss = mae.function(&t, &z).unwrap();

        assert_eq!(loss, 1.5);
    }

    #[test]
    fn mae_function_2d() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let z = Tensor::new(vec![2, 2], vec![2.0, 3.0, 4.0, 5.0]).unwrap();

        let mae = MeanAbsoluteError;
        let loss = mae.function(&t, &z).unwrap();

        assert_eq!(loss, 1.0);
    }

    #[test]
    fn mae_function_negative_values() {
        let t = Tensor::new(vec![3], vec![-1.0, -2.0, -3.0]).unwrap();
        let z = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        let mae = MeanAbsoluteError;
        let loss = mae.function(&t, &z).unwrap();

        assert_eq!(loss, 4.0);
    }

    #[test]
    fn mae_function_mixed_values() {
        let t = Tensor::new(vec![4], vec![-2.0, -1.0, 1.0, 2.0]).unwrap();
        let z = Tensor::new(vec![4], vec![2.0, 1.0, -1.0, -2.0]).unwrap();

        let mae = MeanAbsoluteError;
        let loss = mae.function(&t, &z).unwrap();

        assert_eq!(loss, 3.0);
    }

    #[test]
    fn mae_function_shape_mismatch() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let z = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();

        let mae = MeanAbsoluteError;
        let result = mae.function(&t, &z);

        assert!(result.is_err());
    }

    #[test]
    fn mae_derivative_perfect_prediction() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let z = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        let mae = MeanAbsoluteError;
        let grad = mae.derivative(&t, &z).unwrap();

        assert!(grad.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn mae_derivative_simple_case() {
        let t = Tensor::new(vec![2], vec![3.0, 5.0]).unwrap();
        let z = Tensor::new(vec![2], vec![2.0, 3.0]).unwrap();

        let mae = MeanAbsoluteError;
        let grad = mae.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2]);
        assert!((grad.data[0] - (-0.5)).abs() < EPSILON);
        assert!((grad.data[1] - (-0.5)).abs() < EPSILON);
    }

    #[test]
    fn mae_derivative_negative_diff() {
        let t = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let z = Tensor::new(vec![2], vec![3.0, 4.0]).unwrap();

        let mae = MeanAbsoluteError;
        let grad = mae.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2]);
        assert!((grad.data[0] - 0.5).abs() < EPSILON);
        assert!((grad.data[1] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn mae_derivative_shape_mismatch() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let z = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();

        let mae = MeanAbsoluteError;
        let result = mae.derivative(&t, &z);

        assert!(result.is_err());
    }

    #[test]
    fn mae_derivative_preserves_shape() {
        let t = Tensor::new(vec![3, 4], vec![5.0; 12]).unwrap();
        let z = Tensor::new(vec![3, 4], vec![2.0; 12]).unwrap();

        let mae = MeanAbsoluteError;
        let grad = mae.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, t.shape);
        assert_eq!(grad.dimension, t.dimension);
    }
}

#[cfg(test)]
mod bce_tests {
    use super::*;

    #[test]
    fn bce_function_perfect_prediction() {
        let t = Tensor::new(vec![3], vec![1.0, 0.0, 1.0]).unwrap();
        let z = Tensor::new(vec![3], vec![1.0, 0.0, 1.0]).unwrap();

        let bce = BinaryCrossEntropy;
        let loss = bce.function(&t, &z).unwrap();

        assert!(loss < 1e-5);
    }

    #[test]
    fn bce_function_simple_case() {
        let t = Tensor::new(vec![2], vec![1.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2], vec![0.9, 0.1]).unwrap();

        let bce = BinaryCrossEntropy;
        let loss = bce.function(&t, &z).unwrap();

        assert!(loss > 0.0);
        assert!(loss < 0.2);
    }

    #[test]
    fn bce_function_2d() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2, 2], vec![0.8, 0.2, 0.9, 0.1]).unwrap();

        let bce = BinaryCrossEntropy;
        let loss = bce.function(&t, &z).unwrap();

        assert!(loss > 0.0);
    }

    #[test]
    fn bce_function_worst_case() {
        let t = Tensor::new(vec![2], vec![1.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2], vec![0.0, 1.0]).unwrap();

        let bce = BinaryCrossEntropy;
        let loss = bce.function(&t, &z).unwrap();

        assert!(loss > 10.0);
    }

    #[test]
    fn bce_function_numerical_stability() {
        let t = Tensor::new(vec![2], vec![1.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2], vec![1.0, 0.0]).unwrap();

        let bce = BinaryCrossEntropy;
        let loss = bce.function(&t, &z).unwrap();

        assert!(loss.is_finite());
    }

    #[test]
    fn bce_function_shape_mismatch() {
        let t = Tensor::new(vec![3], vec![1.0, 0.0, 1.0]).unwrap();
        let z = Tensor::new(vec![2], vec![0.5, 0.5]).unwrap();

        let bce = BinaryCrossEntropy;
        let result = bce.function(&t, &z);

        assert!(result.is_err());
    }

    #[test]
    fn bce_derivative_simple_case() {
        let t = Tensor::new(vec![2], vec![1.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2], vec![0.9, 0.1]).unwrap();

        let bce = BinaryCrossEntropy;
        let grad = bce.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2]);
        assert!(grad.data[0] < 0.0);
        assert!(grad.data[1] > 0.0);
    }

    #[test]
    fn bce_derivative_2d() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2, 2], vec![0.8, 0.2, 0.9, 0.1]).unwrap();

        let bce = BinaryCrossEntropy;
        let grad = bce.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2, 2]);
    }

    #[test]
    fn bce_derivative_numerical_stability() {
        let t = Tensor::new(vec![2], vec![1.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2], vec![1.0, 0.0]).unwrap();

        let bce = BinaryCrossEntropy;
        let grad = bce.derivative(&t, &z).unwrap();

        assert!(grad.data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn bce_derivative_shape_mismatch() {
        let t = Tensor::new(vec![3], vec![1.0, 0.0, 1.0]).unwrap();
        let z = Tensor::new(vec![2], vec![0.5, 0.5]).unwrap();

        let bce = BinaryCrossEntropy;
        let result = bce.derivative(&t, &z);

        assert!(result.is_err());
    }

    #[test]
    fn bce_derivative_preserves_shape() {
        let t = Tensor::new(vec![3, 4], vec![1.0; 12]).unwrap();
        let z = Tensor::new(vec![3, 4], vec![0.5; 12]).unwrap();

        let bce = BinaryCrossEntropy;
        let grad = bce.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, t.shape);
        assert_eq!(grad.dimension, t.dimension);
    }
}

#[cfg(test)]
mod cce_tests {
    use super::*;

    #[test]
    fn cce_function_perfect_prediction() {
        let t = Tensor::new(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let z = Tensor::new(vec![3], vec![1.0, 0.0, 0.0]).unwrap();

        let cce = CategoricalCrossEntropy;
        let loss = cce.function(&t, &z).unwrap();

        assert!(loss < 1e-5);
    }

    #[test]
    fn cce_function_simple_case() {
        let t = Tensor::new(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let z = Tensor::new(vec![3], vec![0.9, 0.05, 0.05]).unwrap();

        let cce = CategoricalCrossEntropy;
        let loss = cce.function(&t, &z).unwrap();

        assert!(loss > 0.0);
        assert!(loss < 0.1);
    }

    #[test]
    fn cce_function_2d() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2, 3], vec![0.8, 0.1, 0.1, 0.1, 0.8, 0.1]).unwrap();

        let cce = CategoricalCrossEntropy;
        let loss = cce.function(&t, &z).unwrap();

        assert!(loss > 0.0);
    }

    #[test]
    fn cce_function_worst_case() {
        let t = Tensor::new(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let z = Tensor::new(vec![3], vec![0.0, 0.5, 0.5]).unwrap();

        let cce = CategoricalCrossEntropy;
        let loss = cce.function(&t, &z).unwrap();

        assert!(loss > 5.0);
    }

    #[test]
    fn cce_function_numerical_stability() {
        let t = Tensor::new(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let z = Tensor::new(vec![3], vec![1.0, 0.0, 0.0]).unwrap();

        let cce = CategoricalCrossEntropy;
        let loss = cce.function(&t, &z).unwrap();

        assert!(loss.is_finite());
    }

    #[test]
    fn cce_function_shape_mismatch() {
        let t = Tensor::new(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2], vec![0.5, 0.5]).unwrap();

        let cce = CategoricalCrossEntropy;
        let result = cce.function(&t, &z);

        assert!(result.is_err());
    }

    #[test]
    fn cce_derivative_simple_case() {
        let t = Tensor::new(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let z = Tensor::new(vec![3], vec![0.9, 0.05, 0.05]).unwrap();

        let cce = CategoricalCrossEntropy;
        let grad = cce.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![3]);
        assert!(grad.data[0] < 0.0);
        assert_eq!(grad.data[1], 0.0);
        assert_eq!(grad.data[2], 0.0);
    }

    #[test]
    fn cce_derivative_2d() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2, 3], vec![0.8, 0.1, 0.1, 0.1, 0.8, 0.1]).unwrap();

        let cce = CategoricalCrossEntropy;
        let grad = cce.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2, 3]);
    }

    #[test]
    fn cce_derivative_numerical_stability() {
        let t = Tensor::new(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let z = Tensor::new(vec![3], vec![1.0, 0.0, 0.0]).unwrap();

        let cce = CategoricalCrossEntropy;
        let grad = cce.derivative(&t, &z).unwrap();

        assert!(grad.data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn cce_derivative_shape_mismatch() {
        let t = Tensor::new(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2], vec![0.5, 0.5]).unwrap();

        let cce = CategoricalCrossEntropy;
        let result = cce.derivative(&t, &z);

        assert!(result.is_err());
    }

    #[test]
    fn cce_derivative_preserves_shape() {
        let t = Tensor::new(
            vec![3, 4],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let z = Tensor::new(vec![3, 4], vec![0.5; 12]).unwrap();

        let cce = CategoricalCrossEntropy;
        let grad = cce.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, t.shape);
        assert_eq!(grad.dimension, t.dimension);
    }
}

#[cfg(test)]
mod huber_tests {
    use super::*;

    #[test]
    fn huber_function_perfect_prediction() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let z = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        let huber = HuberLoss { delta: 1.0 };
        let loss = huber.function(&t, &z).unwrap();

        assert_eq!(loss, 0.0);
    }

    #[test]
    fn huber_function_small_error() {
        let t = Tensor::new(vec![2], vec![3.0, 5.0]).unwrap();
        let z = Tensor::new(vec![2], vec![2.5, 4.5]).unwrap();

        let huber = HuberLoss { delta: 1.0 };
        let loss = huber.function(&t, &z).unwrap();

        assert!((loss - 0.125).abs() < EPSILON);
    }

    #[test]
    fn huber_function_large_error() {
        let t = Tensor::new(vec![2], vec![0.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2], vec![3.0, 4.0]).unwrap();

        let huber = HuberLoss { delta: 1.0 };
        let loss = huber.function(&t, &z).unwrap();

        assert_eq!(loss, 3.0);
    }

    #[test]
    fn huber_function_mixed_errors() {
        let t = Tensor::new(vec![3], vec![0.0, 0.0, 0.0]).unwrap();
        let z = Tensor::new(vec![3], vec![0.5, 2.0, 5.0]).unwrap();

        let huber = HuberLoss { delta: 1.0 };
        let loss = huber.function(&t, &z).unwrap();

        let expected = (0.125 + 1.5 + 4.5) / 3.0;
        assert!((loss - expected).abs() < EPSILON);
    }

    #[test]
    fn huber_function_different_deltas() {
        let t = Tensor::new(vec![2], vec![0.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2], vec![2.0, 2.0]).unwrap();

        let huber1 = HuberLoss { delta: 1.0 };
        let loss1 = huber1.function(&t, &z).unwrap();

        let huber2 = HuberLoss { delta: 3.0 };
        let loss2 = huber2.function(&t, &z).unwrap();

        assert!(loss2 > loss1);
    }

    #[test]
    fn huber_function_2d() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let z = Tensor::new(vec![2, 2], vec![2.0, 3.0, 4.0, 5.0]).unwrap();

        let huber = HuberLoss { delta: 1.0 };
        let loss = huber.function(&t, &z).unwrap();

        assert_eq!(loss, 0.5);
    }

    #[test]
    fn huber_function_shape_mismatch() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let z = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();

        let huber = HuberLoss { delta: 1.0 };
        let result = huber.function(&t, &z);

        assert!(result.is_err());
    }

    #[test]
    fn huber_derivative_perfect_prediction() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let z = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        let huber = HuberLoss { delta: 1.0 };
        let grad = huber.derivative(&t, &z).unwrap();

        assert_eq!(grad.data.as_slice().unwrap(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn huber_derivative_small_error() {
        let t = Tensor::new(vec![2], vec![3.0, 5.0]).unwrap();
        let z = Tensor::new(vec![2], vec![2.5, 4.5]).unwrap();

        let huber = HuberLoss { delta: 1.0 };
        let grad = huber.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2]);
        for &val in grad.data.iter() {
            assert!(val.abs() < 1.0);
        }
    }

    #[test]
    fn huber_derivative_large_error() {
        let t = Tensor::new(vec![2], vec![0.0, 0.0]).unwrap();
        let z = Tensor::new(vec![2], vec![3.0, 4.0]).unwrap();

        let huber = HuberLoss { delta: 1.0 };
        let grad = huber.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2]);
        for &val in grad.data.iter() {
            assert!((val.abs() - 0.5).abs() < EPSILON);
        }
    }

    #[test]
    fn huber_derivative_2d() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let z = Tensor::new(vec![2, 2], vec![2.0, 3.0, 4.0, 5.0]).unwrap();

        let huber = HuberLoss { delta: 1.0 };
        let grad = huber.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2, 2]);
    }

    #[test]
    fn huber_derivative_shape_mismatch() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let z = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();

        let huber = HuberLoss { delta: 1.0 };
        let result = huber.derivative(&t, &z);

        assert!(result.is_err());
    }

    #[test]
    fn huber_derivative_preserves_shape() {
        let t = Tensor::new(vec![3, 4], vec![1.0; 12]).unwrap();
        let z = Tensor::new(vec![3, 4], vec![2.0; 12]).unwrap();

        let huber = HuberLoss { delta: 1.0 };
        let grad = huber.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, t.shape);
        assert_eq!(grad.dimension, t.dimension);
    }
}

#[cfg(test)]
mod hinge_tests {
    use super::*;

    #[test]
    fn hinge_function_perfect_prediction() {
        let t = Tensor::new(vec![2], vec![1.0, -1.0]).unwrap();
        let z = Tensor::new(vec![2], vec![2.0, -2.0]).unwrap();

        let hinge = HingeLoss;
        let loss = hinge.function(&t, &z).unwrap();

        assert_eq!(loss, 0.0);
    }

    #[test]
    fn hinge_function_simple_case() {
        let t = Tensor::new(vec![2], vec![1.0, -1.0]).unwrap();
        let z = Tensor::new(vec![2], vec![0.5, -0.5]).unwrap();

        let hinge = HingeLoss;
        let loss = hinge.function(&t, &z).unwrap();

        assert_eq!(loss, 0.5);
    }

    #[test]
    fn hinge_function_worst_case() {
        let t = Tensor::new(vec![2], vec![1.0, -1.0]).unwrap();
        let z = Tensor::new(vec![2], vec![-2.0, 2.0]).unwrap();

        let hinge = HingeLoss;
        let loss = hinge.function(&t, &z).unwrap();

        assert_eq!(loss, 3.0);
    }

    #[test]
    fn hinge_function_mixed_values() {
        let t = Tensor::new(vec![3], vec![1.0, 1.0, -1.0]).unwrap();
        let z = Tensor::new(vec![3], vec![2.0, 0.5, 0.5]).unwrap();

        let hinge = HingeLoss;
        let loss = hinge.function(&t, &z).unwrap();

        let expected = (0.0 + 0.5 + 1.5) / 3.0;
        assert!((loss - expected).abs() < EPSILON);
    }

    #[test]
    fn hinge_function_2d() {
        let t = Tensor::new(vec![2, 2], vec![1.0, -1.0, 1.0, -1.0]).unwrap();
        let z = Tensor::new(vec![2, 2], vec![2.0, -2.0, 0.5, -0.5]).unwrap();

        let hinge = HingeLoss;
        let loss = hinge.function(&t, &z).unwrap();

        assert_eq!(loss, 0.25);
    }

    #[test]
    fn hinge_function_shape_mismatch() {
        let t = Tensor::new(vec![3], vec![1.0, -1.0, 1.0]).unwrap();
        let z = Tensor::new(vec![2], vec![1.0, -1.0]).unwrap();

        let hinge = HingeLoss;
        let result = hinge.function(&t, &z);

        assert!(result.is_err());
    }

    #[test]
    fn hinge_derivative_perfect_prediction() {
        let t = Tensor::new(vec![2], vec![1.0, -1.0]).unwrap();
        let z = Tensor::new(vec![2], vec![2.0, -2.0]).unwrap();

        let hinge = HingeLoss;
        let grad = hinge.derivative(&t, &z).unwrap();

        assert_eq!(grad.data.as_slice().unwrap(), &[0.0, 0.0]);
    }

    #[test]
    fn hinge_derivative_simple_case() {
        let t = Tensor::new(vec![2], vec![1.0, -1.0]).unwrap();
        let z = Tensor::new(vec![2], vec![0.5, -0.5]).unwrap();

        let hinge = HingeLoss;
        let grad = hinge.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2]);
        assert!((grad.data[0] - (-0.5)).abs() < EPSILON);
        assert!((grad.data[1] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn hinge_derivative_worst_case() {
        let t = Tensor::new(vec![2], vec![1.0, -1.0]).unwrap();
        let z = Tensor::new(vec![2], vec![-2.0, 2.0]).unwrap();

        let hinge = HingeLoss;
        let grad = hinge.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2]);
        assert!((grad.data[0] - (-0.5)).abs() < EPSILON);
        assert!((grad.data[1] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn hinge_derivative_2d() {
        let t = Tensor::new(vec![2, 2], vec![1.0, -1.0, 1.0, -1.0]).unwrap();
        let z = Tensor::new(vec![2, 2], vec![2.0, -2.0, 0.5, -0.5]).unwrap();

        let hinge = HingeLoss;
        let grad = hinge.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, vec![2, 2]);
    }

    #[test]
    fn hinge_derivative_shape_mismatch() {
        let t = Tensor::new(vec![3], vec![1.0, -1.0, 1.0]).unwrap();
        let z = Tensor::new(vec![2], vec![1.0, -1.0]).unwrap();

        let hinge = HingeLoss;
        let result = hinge.derivative(&t, &z);

        assert!(result.is_err());
    }

    #[test]
    fn hinge_derivative_preserves_shape() {
        let t = Tensor::new(vec![3, 4], vec![1.0; 12]).unwrap();
        let z = Tensor::new(vec![3, 4], vec![0.5; 12]).unwrap();

        let hinge = HingeLoss;
        let grad = hinge.derivative(&t, &z).unwrap();

        assert_eq!(grad.shape, t.shape);
        assert_eq!(grad.dimension, t.dimension);
    }
}
