//! Test suite for activation functions
//!
//! Tests verify both the forward pass (function) and backward pass (derivative)
//! for all activation functions: Relu, Sigmoid, Tanh, Softmax, Linear.

use crate::math::activations::{ActivationFunction, Relu, Sigmoid, Softmax, Tanh};
use crate::tensor::Tensor;

// =============================================================================
// RELU ACTIVATION
// =============================================================================

#[cfg(test)]
mod relu_tests {
    use super::*;

    #[test]
    fn relu_function_positive_values() {
        let t = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let relu = Relu;
        let result = relu.function(&t).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn relu_function_negative_values() {
        let t = Tensor::new(vec![4], vec![-1.0, -2.0, -3.0, -4.0]).unwrap();
        let relu = Relu;
        let result = relu.function(&t).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn relu_function_mixed_values() {
        let t = Tensor::new(vec![4], vec![-2.0, -1.0, 1.0, 2.0]).unwrap();
        let relu = Relu;
        let result = relu.function(&t).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn relu_function_zero() {
        let t = Tensor::new(vec![3], vec![0.0, 0.0, 0.0]).unwrap();
        let relu = Relu;
        let result = relu.function(&t).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn relu_function_2d() {
        let t = Tensor::new(vec![2, 3], vec![-1.0, 0.0, 1.0, -2.0, 2.0, 3.0]).unwrap();
        let relu = Relu;
        let result = relu.function(&t).unwrap();

        assert_eq!(
            result.data.as_slice().unwrap(),
            &[0.0, 0.0, 1.0, 0.0, 2.0, 3.0]
        );
    }

    #[test]
    fn relu_derivative_positive_input() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let relu = Relu;
        let result = relu.derivative(&t).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn relu_derivative_negative_input() {
        let t = Tensor::new(vec![3], vec![-1.0, -2.0, -3.0]).unwrap();
        let relu = Relu;
        let result = relu.derivative(&t).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn relu_derivative_at_zero() {
        let t = Tensor::new(vec![3], vec![0.0, 0.0, 0.0]).unwrap();
        let relu = Relu;
        let result = relu.derivative(&t).unwrap();

        // Derivative at zero is defined as 0
        assert_eq!(result.data.as_slice().unwrap(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn relu_derivative_mixed() {
        let t = Tensor::new(vec![4], vec![-2.0, 0.0, 1.0, 5.0]).unwrap();
        let relu = Relu;
        let result = relu.derivative(&t).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn relu_derivative_preserves_shape() {
        let t = Tensor::new(vec![2, 3], vec![1.0; 6]).unwrap();
        let relu = Relu;
        let result = relu.derivative(&t).unwrap();

        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.dimension, 2);
    }
}

// =============================================================================
// SIGMOID ACTIVATION
// =============================================================================

#[cfg(test)]
mod sigmoid_tests {
    use super::*;

    #[test]
    fn sigmoid_function_zero() {
        let t = Tensor::new(vec![1], vec![0.0]).unwrap();
        let sigmoid = Sigmoid;
        let result = sigmoid.function(&t).unwrap();

        // sigmoid(0) = 0.5
        assert!((result.data[[0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_function_positive() {
        let t = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let sigmoid = Sigmoid;
        let result = sigmoid.function(&t).unwrap();

        // sigmoid(1) ≈ 0.731, sigmoid(2) ≈ 0.881
        assert!(result.data[[0]] > 0.7 && result.data[[0]] < 0.75);
        assert!(result.data[[1]] > 0.88 && result.data[[1]] < 0.89);
    }

    #[test]
    fn sigmoid_function_negative() {
        let t = Tensor::new(vec![2], vec![-1.0, -2.0]).unwrap();
        let sigmoid = Sigmoid;
        let result = sigmoid.function(&t).unwrap();

        // sigmoid(-1) ≈ 0.269, sigmoid(-2) ≈ 0.119
        assert!(result.data[[0]] > 0.26 && result.data[[0]] < 0.27);
        assert!(result.data[[1]] > 0.11 && result.data[[1]] < 0.12);
    }

    #[test]
    fn sigmoid_function_large_positive() {
        let t = Tensor::new(vec![1], vec![10.0]).unwrap();
        let sigmoid = Sigmoid;
        let result = sigmoid.function(&t).unwrap();

        // sigmoid(10) ≈ 1.0
        assert!(result.data[[0]] > 0.9999);
    }

    #[test]
    fn sigmoid_function_large_negative() {
        let t = Tensor::new(vec![1], vec![-10.0]).unwrap();
        let sigmoid = Sigmoid;
        let result = sigmoid.function(&t).unwrap();

        // sigmoid(-10) ≈ 0.0
        assert!(result.data[[0]] < 0.0001);
    }

    #[test]
    fn sigmoid_function_2d() {
        let t = Tensor::new(vec![2, 2], vec![0.0, 1.0, -1.0, 2.0]).unwrap();
        let sigmoid = Sigmoid;
        let result = sigmoid.function(&t).unwrap();

        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.dimension, 2);
        // All values should be in (0, 1)
        assert!(result.data.iter().all(|&x| x > 0.0 && x < 1.0));
    }

    #[test]
    fn sigmoid_derivative_at_zero() {
        let t = Tensor::new(vec![1], vec![0.5]).unwrap();
        let sigmoid = Sigmoid;
        let result = sigmoid.derivative(&t).unwrap();

        // derivative at sigmoid(0)=0.5: 0.5 * (1 - 0.5) = 0.25
        assert!((result.data[[0]] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_derivative_input_is_sigmoid_output() {
        // Note: derivative expects sigmoid OUTPUT, not input
        let t = Tensor::new(vec![3], vec![0.1, 0.5, 0.9]).unwrap();
        let sigmoid = Sigmoid;
        let result = sigmoid.derivative(&t).unwrap();

        // For sigmoid output s, derivative = s * (1 - s)
        assert!((result.data[[0]] - 0.1 * 0.9).abs() < 1e-6);
        assert!((result.data[[1]] - 0.5 * 0.5).abs() < 1e-6);
        assert!((result.data[[2]] - 0.9 * 0.1).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_derivative_edge_values() {
        let t = Tensor::new(vec![2], vec![0.0, 1.0]).unwrap();
        let sigmoid = Sigmoid;
        let result = sigmoid.derivative(&t).unwrap();

        // derivative(0) = 0, derivative(1) = 0
        assert_eq!(result.data[[0]], 0.0);
        assert_eq!(result.data[[1]], 0.0);
    }

    #[test]
    fn sigmoid_derivative_preserves_shape() {
        let t = Tensor::new(vec![2, 3], vec![0.5; 6]).unwrap();
        let sigmoid = Sigmoid;
        let result = sigmoid.derivative(&t).unwrap();

        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.dimension, 2);
    }
}

// =============================================================================
// TANH ACTIVATION
// =============================================================================

#[cfg(test)]
mod tanh_tests {
    use super::*;

    #[test]
    fn tanh_function_zero() {
        let t = Tensor::new(vec![1], vec![0.0]).unwrap();
        let tanh = Tanh;
        let result = tanh.function(&t).unwrap();

        assert_eq!(result.data[[0]], 0.0);
    }

    #[test]
    fn tanh_function_positive() {
        let t = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let tanh = Tanh;
        let result = tanh.function(&t).unwrap();

        // tanh(1) ≈ 0.762, tanh(2) ≈ 0.964
        assert!((result.data[[0]] - 0.762).abs() < 0.001);
        assert!((result.data[[1]] - 0.964).abs() < 0.001);
    }

    #[test]
    fn tanh_function_negative() {
        let t = Tensor::new(vec![2], vec![-1.0, -2.0]).unwrap();
        let tanh = Tanh;
        let result = tanh.function(&t).unwrap();

        // tanh is antisymmetric
        assert!((result.data[[0]] + 0.762).abs() < 0.001);
        assert!((result.data[[1]] + 0.964).abs() < 0.001);
    }

    #[test]
    fn tanh_function_large_positive() {
        let t = Tensor::new(vec![1], vec![10.0]).unwrap();
        let tanh = Tanh;
        let result = tanh.function(&t).unwrap();

        // tanh(10) ≈ 1.0
        assert!((result.data[[0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tanh_function_large_negative() {
        let t = Tensor::new(vec![1], vec![-10.0]).unwrap();
        let tanh = Tanh;
        let result = tanh.function(&t).unwrap();

        // tanh(-10) ≈ -1.0
        assert!((result.data[[0]] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn tanh_function_range() {
        let t = Tensor::random(vec![100], -10.0..10.0);
        let tanh = Tanh;
        let result = tanh.function(&t).unwrap();

        assert!(result.data.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn tanh_function_2d() {
        let t = Tensor::new(vec![2, 2], vec![0.0, 1.0, -1.0, 2.0]).unwrap();
        let tanh = Tanh;
        let result = tanh.function(&t).unwrap();

        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.dimension, 2);
    }

    #[test]
    fn tanh_derivative_at_zero() {
        let t = Tensor::new(vec![1], vec![0.0]).unwrap();
        let tanh = Tanh;
        let result = tanh.derivative(&t).unwrap();

        // tanh'(0) = 1 - 0^2 = 1
        assert_eq!(result.data[[0]], 1.0);
    }

    #[test]
    fn tanh_derivative_at_extremes() {
        let t = Tensor::new(vec![2], vec![10.0, -10.0]).unwrap();
        let tanh = Tanh;
        let result = tanh.derivative(&t).unwrap();

        // tanh(±10) ≈ ±1, derivative = 1 - 1^2 ≈ 0
        assert!(result.data[[0]] < 1e-5);
        assert!(result.data[[1]] < 1e-5);
    }

    #[test]
    fn tanh_derivative_formula() {
        // derivative of tanh(x) = 1 - tanh(x)^2
        let t = Tensor::new(vec![3], vec![0.5, 1.0, 1.5]).unwrap();
        let tanh = Tanh;
        let result = tanh.derivative(&t).unwrap();

        for (i, &x) in t.data.iter().enumerate() {
            let expected = 1.0 - x.tanh().powf(2.0);
            assert!((result.data[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn tanh_derivative_preserves_shape() {
        let t = Tensor::new(vec![2, 3], vec![0.5; 6]).unwrap();
        let tanh = Tanh;
        let result = tanh.derivative(&t).unwrap();

        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.dimension, 2);
    }
}

// =============================================================================
// SOFTMAX ACTIVATION
// =============================================================================

#[cfg(test)]
mod softmax_tests {
    use super::*;

    #[test]
    fn softmax_function_1d() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let softmax = Softmax;
        let result = softmax.function(&t).unwrap();

        // Output should sum to 1
        let sum: f32 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // All values should be positive
        assert!(result.data.iter().all(|&x| x > 0.0));

        // Larger inputs should have larger outputs
        assert!(result.data[[2]] > result.data[[1]]);
        assert!(result.data[[1]] > result.data[[0]]);
    }

    #[test]
    fn softmax_function_1d_equal_values() {
        let t = Tensor::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let softmax = Softmax;
        let result = softmax.function(&t).unwrap();

        // All outputs should be equal to 0.25
        for &x in result.data.iter() {
            assert!((x - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_function_1d_large_values() {
        // Test numerical stability with large values
        let t = Tensor::new(vec![3], vec![1000.0, 1001.0, 1002.0]).unwrap();
        let softmax = Softmax;
        let result = softmax.function(&t).unwrap();

        // Should still sum to 1 and not overflow
        let sum: f32 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result.data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn softmax_function_2d() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let softmax = Softmax;
        let result = softmax.function(&t).unwrap();

        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.dimension, 2);

        // Each row should sum to 1
        let row1_sum: f32 = (0..3).map(|i| result.data[[0, i]]).sum();
        let row2_sum: f32 = (0..3).map(|i| result.data[[1, i]]).sum();

        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_function_fails_on_3d() {
        let t = Tensor::new(vec![2, 2, 2], vec![1.0; 8]).unwrap();
        let softmax = Softmax;
        let result = softmax.function(&t);

        assert!(result.is_err());
    }

    #[test]
    fn softmax_derivative_1d_returns_jacobian() {
        let t = Tensor::new(vec![3], vec![0.2, 0.5, 0.3]).unwrap();
        let softmax = Softmax;
        let result = softmax.derivative(&t).unwrap();

        // Derivative of softmax is the Jacobian matrix (n x n)
        assert_eq!(result.dimension, 2);
        assert_eq!(result.shape, vec![3, 3]);

        // Verify Jacobian formula: J[i,j] = s[i] * (δ[i,j] - s[j])
        // where s is softmax output
        let s = softmax.function(&t).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j {
                    s.data[i] * (1.0 - s.data[j])
                } else {
                    -s.data[i] * s.data[j]
                };
                assert!((result.data[[i, j]] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn softmax_derivative_2d_returns_3d_jacobians() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let softmax = Softmax;
        let result = softmax.derivative(&t).unwrap();

        // Derivative shape should be [batch_size, n, n]
        assert_eq!(result.dimension, 3);
        assert_eq!(result.shape, vec![2, 3, 3]);
    }

    #[test]
    fn softmax_derivative_fails_on_3d() {
        let t = Tensor::new(vec![2, 2, 2], vec![1.0; 8]).unwrap();
        let softmax = Softmax;
        let result = softmax.derivative(&t);

        assert!(result.is_err());
    }
}
