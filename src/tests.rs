#[cfg(test)]
mod test {
    use crate::structures::tensor::Tensor;
    use crate::structures::layer::Layer;
    use crate::functions::activation::{Activation, ActivationFunction, Relu, Sigmoid};
    use crate::functions::cost::{CostFunction, MeanSquaredError, BinaryCrossEntropy};
    use crate::utils::weights_biases::{random_weights, random_biases};
    use ndarray::{Array, ArrayD};
    // Tensor Tests
    #[test]
    fn test_tensor_creation() {
        // Test creating a tensor with specific shape and content
        let shape = vec![2, 3];
        let content = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(shape.clone(), content.clone());
        
        assert_eq!(tensor.dimension, 2);
        assert_eq!(tensor.shape, shape);
        
        // Test the data content
        for (i, &val) in tensor.data.iter().enumerate() {
            assert_eq!(content[i], val);
        }
    }

    #[test]
    fn test_tensor_zeros() {
        // Test creating a tensor filled with zeros
        let shape = vec![2, 3];
        let tensor = Tensor::zeros(shape.clone());
        
        assert_eq!(tensor.dimension, 2);
        assert_eq!(tensor.shape, shape);
        
        // Test that all values are zero
        for i in tensor.data.iter() {
            assert_eq!(*i, 0.0);
        }
    }

    #[test]
    fn test_tensor_dot_vector_vector() {
        // Test dot product between two vectors
        let t1 = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let t2 = Tensor::new(vec![3], vec![4.0, 5.0, 6.0]);
        
        let result = t1.dot(&t2).unwrap();
        
        // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result.dimension, 0);
        assert_eq!(result.shape, Vec::<usize>::new());
        assert_eq!(result.data, ArrayD::from_elem(Vec::<usize>::new(), 32.0));
    }

    #[test]
    fn test_tensor_dot_vector_matrix() {
        // Test dot product between a vector and a matrix
        let t1 = Tensor::new(vec![2], vec![1.0, 2.0]);
        let t2 = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        let result = t1.dot(&t2).unwrap();
        
        // Expected: [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6] = [9, 12, 15]
        assert_eq!(result.dimension, 1);
        assert_eq!(result.shape, vec![3]);
        assert_eq!(result.data[0], 9.0);
        assert_eq!(result.data[1], 12.0);
        assert_eq!(result.data[2], 15.0);
    }

    #[test]
    fn test_tensor_operations() {
        // Test tensor operations
        let t1 = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let t2 = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        
        // Test tensor sum
        let sum_result = t1.tensor_sum(&t2).unwrap();
        assert_eq!(sum_result.data[[0, 0]], 6.0);
        assert_eq!(sum_result.data[[0, 1]], 8.0);
        assert_eq!(sum_result.data[[1, 0]], 10.0);
        assert_eq!(sum_result.data[[1, 1]], 12.0);
        
        // Test tensor subtraction
        let sub_result = t2.tensor_subtraction(&t1).unwrap();
        assert_eq!(sub_result.data[[0, 0]], 4.0);
        assert_eq!(sub_result.data[[0, 1]], 4.0);
        assert_eq!(sub_result.data[[1, 0]], 4.0);
        assert_eq!(sub_result.data[[1, 1]], 4.0);
        
        // Test tensor multiplication
        let mul_result = t1.tensor_multiplication(&t2).unwrap();
        assert_eq!(mul_result.data[[0, 0]], 5.0);
        assert_eq!(mul_result.data[[0, 1]], 12.0);
        assert_eq!(mul_result.data[[1, 0]], 21.0);
        assert_eq!(mul_result.data[[1, 1]], 32.0);
        
        // Test tensor division
        let div_t1 = Tensor::new(vec![2, 2], vec![10.0, 12.0, 9.0, 8.0]);
        let div_t2 = Tensor::new(vec![2, 2], vec![2.0, 3.0, 3.0, 2.0]);
        let div_result = div_t1.tensor_division(&div_t2).unwrap();
        assert_eq!(div_result.data[[0, 0]], 5.0);
        assert_eq!(div_result.data[[0, 1]], 4.0);
        assert_eq!(div_result.data[[1, 0]], 3.0);
        assert_eq!(div_result.data[[1, 1]], 4.0);
    }

    #[test]
    fn test_scalar_operations() {
        // Test scalar operations
        let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        
        // Test scalar addition
        let sum_result = t.scalar_sum(2.0).unwrap();
        assert_eq!(sum_result.data[[0, 0]], 3.0);
        assert_eq!(sum_result.data[[0, 1]], 4.0);
        assert_eq!(sum_result.data[[1, 0]], 5.0);
        assert_eq!(sum_result.data[[1, 1]], 6.0);
        
        // Test scalar subtraction
        let sub_result = t.scalar_subtraction(1.0).unwrap();
        assert_eq!(sub_result.data[[0, 0]], 0.0);
        assert_eq!(sub_result.data[[0, 1]], 1.0);
        assert_eq!(sub_result.data[[1, 0]], 2.0);
        assert_eq!(sub_result.data[[1, 1]], 3.0);
        
        // Test scalar multiplication
        let mul_result = t.scalar_multiplication(3.0).unwrap();
        assert_eq!(mul_result.data[[0, 0]], 3.0);
        assert_eq!(mul_result.data[[0, 1]], 6.0);
        assert_eq!(mul_result.data[[1, 0]], 9.0);
        assert_eq!(mul_result.data[[1, 1]], 12.0);
        
        // Test scalar division
        let div_result = t.scalar_division(2.0).unwrap();
        assert_eq!(div_result.data[[0, 0]], 0.5);
        assert_eq!(div_result.data[[0, 1]], 1.0);
        assert_eq!(div_result.data[[1, 0]], 1.5);
        assert_eq!(div_result.data[[1, 1]], 2.0);
    }

    #[test]
    fn test_transpose() {
        // Test matrix transpose
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        let result = t.transpose().unwrap();
        
        assert_eq!(result.dimension, 2);
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(result.data[[0, 0]], 1.0);
        assert_eq!(result.data[[0, 1]], 4.0);
        assert_eq!(result.data[[1, 0]], 2.0);
        assert_eq!(result.data[[1, 1]], 5.0);
        assert_eq!(result.data[[2, 0]], 3.0);
        assert_eq!(result.data[[2, 1]], 6.0);
    }

    #[test]
    fn test_tensor_reshape() {
        // Test tensor reshape
        let mut t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

        t.reshape(vec![4]);

        assert_eq!(t.dimension, 1);
        assert_eq!(t.shape, vec![4]);
        assert_eq!(t.data[0], 1.0);
        assert_eq!(t.data[1], 2.0);
        assert_eq!(t.data[2], 3.0);
        assert_eq!(t.data[3], 4.0);
    }

    #[test]
    fn test_tensor_flatten() {
        // Test tensor flatten
        let mut t = Tensor::new(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        t.flatten();

        assert_eq!(t.dimension, 1);
        assert_eq!(t.shape, vec![8]);
        assert_eq!(t.data[0], 1.0);
        assert_eq!(t.data[1], 2.0);
        assert_eq!(t.data[2], 3.0);
        assert_eq!(t.data[3], 4.0);
        assert_eq!(t.data[4], 5.0);
        assert_eq!(t.data[5], 6.0);
        assert_eq!(t.data[6], 7.0);
        assert_eq!(t.data[7], 8.0);
        
    }

    #[test]
    fn test_tensor_cat() {
        // Test tensor cat
        let t_1 = Tensor::new(vec![2], vec![1.0, 2.0]);
        let t_2 = Tensor::new(vec![2], vec![3.0, 4.0]);
        let t_3 = Tensor::new(vec![2], vec![5.0, 6.0]);
        let t_4 = Tensor::new(vec![2], vec![7.0, 8.0]);

        let t = t_1.cat(vec![t_2, t_3, t_4], 0).unwrap();

        assert_eq!(t.dimension, 1);
        assert_eq!(t.shape, vec![8]);
        assert_eq!(t.data[0], 1.0);
        assert_eq!(t.data[1], 2.0);
        assert_eq!(t.data[2], 3.0);
        assert_eq!(t.data[3], 4.0);
        assert_eq!(t.data[4], 5.0);
        assert_eq!(t.data[5], 6.0);
        assert_eq!(t.data[6], 7.0);
        assert_eq!(t.data[7], 8.0);
    }

    #[test]
    fn test_tensor_push_row() {
        // Test tensor row
        let mut t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t_2 = Tensor::new(vec![3], vec![7.0, 8.0, 9.0]);

        t.push_row(&t_2);

        assert_eq!(t.dimension, 2);
        assert_eq!(t.shape, vec![3, 3]);
        assert_eq!(t.data[[0, 0]], 1.0);
        assert_eq!(t.data[[0, 1]], 2.0);
        assert_eq!(t.data[[0, 2]], 3.0);
        assert_eq!(t.data[[1, 0]], 4.0);
        assert_eq!(t.data[[1, 1]], 5.0);
        assert_eq!(t.data[[1, 2]], 6.0);
        assert_eq!(t.data[[2, 0]], 7.0);
        assert_eq!(t.data[[2, 1]], 8.0);
        assert_eq!(t.data[[2, 2]], 9.0);
    }

    // Layer Tests
    #[test]
    fn test_layer_creation() {
        // Test creating a layer
        let nodes = 4;
        let input_len = 3;
        let activation = Activation::Relu;
        
        let layer = Layer::new(nodes, input_len, activation);
        
        assert_eq!(layer.nodes, nodes);
        assert_eq!(layer.input.shape, vec![input_len]);
        assert_eq!(layer.output.shape, vec![nodes]);
        assert_eq!(layer.weights.shape, vec![input_len, nodes]);
        assert_eq!(layer.biases.shape, vec![nodes]);
        
        // Check that weights and biases are initialized
        assert!(layer.weights.data.iter().any(|&x| x != 0.0));
        assert!(layer.biases.data.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_layer_forward() {
        // Test forward propagation
        let nodes = 2;
        let input_len = 3;
        let activation = Activation::Relu;
        
        let mut layer = Layer::new(nodes, input_len, activation);
        
        // Set specific weights and biases for predictable output
        let weights_data = Array::from_shape_vec((input_len, nodes), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let biases_data = Array::from_shape_vec(nodes, vec![0.1, 0.2]).unwrap();
        
        layer.weights = Tensor {
            dimension: 2,
            shape: vec![input_len, nodes],
            data: weights_data.into_dyn(),
        };
        
        layer.biases = Tensor {
            dimension: 1,
            shape: vec![nodes],
            data: biases_data.into_dyn(),
        };
        
        // Create input tensor
        let input = Tensor::new(vec![input_len], vec![1.0, 2.0, 3.0]);
        
        // Forward propagation
        let output = layer.forward(input, false).unwrap();
        
        // Expected output calculation:
        // [1.0, 2.0, 3.0] · [0.1, 0.2; 0.3, 0.4; 0.5, 0.6] + [0.1, 0.2]
        // = [1.0*0.1 + 2.0*0.3 + 3.0*0.5, 1.0*0.2 + 2.0*0.4 + 3.0*0.6] + [0.1, 0.2]
        // = [0.1 + 0.6 + 1.5, 0.2 + 0.8 + 1.8] + [0.1, 0.2]
        // = [2.2, 2.8] + [0.1, 0.2]
        // = [2.3, 3.0]
        // After ReLU: [2.3, 3.0] (all positive, so unchanged)
        
        assert_eq!(output.shape, vec![nodes]);
        assert!(f64::abs(output.data[0] - 2.3) < 1e-10);
        assert!(f64::abs(output.data[1] - 3.0) < 1e-10);
    }

    #[test]
    fn test_layer_forward_batch() {
        let mut t_1 = Tensor::new(vec![3], vec![1.0, 1.0, 1.0]);
        let t_2 = Tensor::new(vec![3], vec![2.0, 2.0, 2.0]);
        let t_3 = Tensor::new(vec![3], vec![3.0, 3.0, 3.0]);

        t_1.push_row(&t_2);
        t_1.push_row(&t_3);

        let mut layer = Layer::new(2, t_1.shape[1], Activation::Softmax);

        let output = layer.forward_batch(t_1, true).unwrap();
        println!("{:#?}", output.data);
        println!("{:#?}", layer.output.data);
    }

    // Activation Function Tests
    #[test]
    fn test_relu_activation() {
        // Test ReLU activation function
        let relu = Relu;
        
        // Test positive values
        let mut t_pos = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let result_pos = relu.function(&mut t_pos);
        
        assert_eq!(result_pos.data[0], 1.0);
        assert_eq!(result_pos.data[1], 2.0);
        assert_eq!(result_pos.data[2], 3.0);
        
        // Test negative values
        let mut t_neg = Tensor::new(vec![3], vec![-1.0, -2.0, -3.0]);
        let result_neg = relu.function(&mut t_neg);
        
        assert_eq!(result_neg.data[0], 0.0);
        assert_eq!(result_neg.data[1], 0.0);
        assert_eq!(result_neg.data[2], 0.0);
        
        // Test mixed values
        let mut t_mixed = Tensor::new(vec![4], vec![-1.0, 0.0, 1.0, 2.0]);
        let result_mixed = relu.function(&mut t_mixed);
        
        assert_eq!(result_mixed.data[0], 0.0);
        assert_eq!(result_mixed.data[1], 0.0);
        assert_eq!(result_mixed.data[2], 1.0);
        assert_eq!(result_mixed.data[3], 2.0);
        
        // Test derivative
        let mut t_deriv = Tensor::new(vec![4], vec![-1.0, 0.0, 1.0, 2.0]);
        let result_deriv = relu.derivative(&mut t_deriv);
        
        assert_eq!(result_deriv.data[0], 0.0); // Derivative of ReLU at x < 0 is 0
        assert_eq!(result_deriv.data[1], 0.0); // Derivative of ReLU at x = 0 is 0 (by convention)
        assert_eq!(result_deriv.data[2], 1.0); // Derivative of ReLU at x > 0 is 1
        assert_eq!(result_deriv.data[3], 1.0);
    }

    #[test]
    fn test_sigmoid_activation() {
        // Test Sigmoid activation function
        let sigmoid = Sigmoid;
        
        // Test function
        let mut t = Tensor::new(vec![3], vec![-2.0, 0.0, 2.0]);
        let result = sigmoid.function(&mut t);
        
        // Expected: sigmoid(-2) ≈ 0.119, sigmoid(0) = 0.5, sigmoid(2) ≈ 0.881
        assert!(f64::abs(result.data[0] - 0.119202922) < 1e-8);
        assert!(f64::abs(result.data[1] - 0.5) < 1e-10);
        assert!(f64::abs(result.data[2] - 0.880797078) < 1e-8);
        
        // Test derivative
        // For sigmoid, the derivative is sigmoid(x) * (1 - sigmoid(x))
        let mut t_deriv = Tensor::new(vec![3], vec![0.119202922, 0.5, 0.880797078]);
        let result_deriv = sigmoid.derivative(&mut t_deriv);
        
        // Expected: 0.119202922 * (1 - 0.119202922) ≈ 0.105, 0.5 * (1 - 0.5) = 0.25, 0.880797078 * (1 - 0.880797078) ≈ 0.105
        assert!(f64::abs(result_deriv.data[0] - 0.105) < 1e-3);
        assert!(f64::abs(result_deriv.data[1] - 0.25) < 1e-10);
        assert!(f64::abs(result_deriv.data[2] - 0.105) < 1e-3);
    }

    // Cost Function Tests
    #[test]
    fn test_mean_squared_error() {
        // Test Mean Squared Error cost function
        let mse = MeanSquaredError;
        
        // Create target and predicted tensors
        let t = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let z = Tensor::new(vec![4], vec![1.1, 2.2, 2.9, 4.2]);
        
        // Calculate MSE
        let cost = mse.function(&t, &z);
        
        // Expected: ((1.0-1.1)² + (2.0-2.2)² + (3.0-2.9)² + (4.0-4.2)²) / 4
        // = (0.01 + 0.04 + 0.01 + 0.04) / 4 = 0.1 / 4 = 0.025
        assert!(f64::abs(cost - 0.025) < 1e-10);
        
        // Test derivative
        let derivative = mse.derivative(&t, &z);
        
        // Expected: -2/n * (t - z) = -2/4 * [1.0-1.1, 2.0-2.2, 3.0-2.9, 4.0-4.2]
        // = -0.5 * [-0.1, -0.2, 0.1, -0.2] = [0.05, 0.1, -0.05, 0.1]
        assert!(f64::abs(derivative.data[0] - 0.05) < 1e-10);
        assert!(f64::abs(derivative.data[1] - 0.1) < 1e-10);
        assert!(f64::abs(derivative.data[2] + 0.05) < 1e-10);
        assert!(f64::abs(derivative.data[3] - 0.1) < 1e-10);
    }

    #[test]
    fn test_binary_cross_entropy() {
        // Test Binary Cross-Entropy cost function
        let bce = BinaryCrossEntropy;
        
        // Create target and predicted tensors
        let t = Tensor::new(vec![3], vec![0.0, 1.0, 0.5]);
        let z = Tensor::new(vec![3], vec![0.1, 0.9, 0.5]);
        
        // Calculate BCE
        let cost = bce.function(&t, &z);
        
        // Expected: -1/3 * (0*ln(0.1) + (1-0)*ln(1-0.1) + 1*ln(0.9) + (1-1)*ln(1-0.9) + 0.5*ln(0.5) + (1-0.5)*ln(1-0.5))
        // = -1/3 * (0 + ln(0.9) + ln(0.9) + 0 + 0.5*ln(0.5) + 0.5*ln(0.5))
        // ≈ -1/3 * (2*ln(0.9) + ln(0.5))
        // ≈ -1/3 * (2*(-0.105) + (-0.693))
        // ≈ -1/3 * (-0.21 - 0.693)
        // ≈ -1/3 * (-0.903)
        // ≈ 0.301
        assert!(f64::abs(cost - 0.301) < 1e-3);
    }

    // Random Weights and Biases Tests
    #[test]
    fn test_random_weights() {
        // Test random weights generation
        let nodes_1 = 3;
        let nodes_2 = 2;
        let range = (-1.0, 1.0);
        
        let weights = random_weights(nodes_1, nodes_2, range);
        
        assert_eq!(weights.dimension, 2);
        assert_eq!(weights.shape, vec![nodes_1, nodes_2]);
        
        // Check that weights are within the specified range
        for i in 0..nodes_1 {
            for j in 0..nodes_2 {
                let w = weights.data[[i, j]];
                assert!(w >= range.0 && w <= range.1);
            }
        }
    }

    #[test]
    fn test_random_biases() {
        // Test random biases generation
        let nodes = 4;
        let range = (-1.0, 1.0);
        
        let biases = random_biases(nodes, range);
        
        assert_eq!(biases.dimension, 1);
        assert_eq!(biases.shape, vec![nodes]);
        
        // Check that biases are within the specified range
        for i in 0..nodes {
            let b = biases.data[i];
            assert!(b >= range.0 && b <= range.1);
        }
    }
}
