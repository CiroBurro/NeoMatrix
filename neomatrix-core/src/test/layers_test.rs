use crate::layers::activations::Softmax;
use crate::layers::{dense::Dense, init::Init, Layer};
use crate::tensor::Tensor;
use std::sync::Arc;

#[cfg(test)]
mod dense_tests {
    use super::*;

    #[test]
    fn dense_get_parameters_immediately_after_construction() {
        let layer = Dense::new(10, 5, None, None);
        let params = layer.get_parameters();

        assert!(params.is_some());

        let p = params.unwrap();

        let w_shape = p.weights.lock().unwrap().shape.clone();
        assert_eq!(w_shape, vec![10, 5]);

        let b_shape = p.biases.lock().unwrap().shape.clone();
        assert_eq!(b_shape, vec![5]);

        let wg_shape = p.w_grads.lock().unwrap().shape.clone();
        assert_eq!(wg_shape, vec![10, 5]);

        let bg_shape = p.b_grads.lock().unwrap().shape.clone();
        assert_eq!(bg_shape, vec![5]);
    }

    #[test]
    fn dense_gradients_initialized_to_zero() {
        let layer = Dense::new(3, 2, None, None);
        let params = layer.get_parameters().unwrap();

        let w_grads = params.w_grads.lock().unwrap();
        assert!(
            w_grads.data.iter().all(|&x| x == 0.0),
            "Weight gradients should be initialized to zero"
        );

        let b_grads = params.b_grads.lock().unwrap();
        assert!(
            b_grads.data.iter().all(|&x| x == 0.0),
            "Bias gradients should be initialized to zero"
        );
    }

    #[test]
    fn dense_forward_correct_output_shape() {
        let mut layer = Dense::new(4, 3, Some(Init::Xavier), None);
        let input = Tensor::random(vec![2, 4], -1.0..1.0);

        let output = layer.forward(&input, true).unwrap();

        assert_eq!(output.dimension, 2);
        assert_eq!(output.shape, vec![2, 3]);
    }

    #[test]
    fn dense_forward_training_mode() {
        let mut layer = Dense::new(5, 3, None, None);
        let input = Tensor::zeros(vec![2, 5]);

        let result = layer.forward(&input, true);

        assert!(
            result.is_ok(),
            "Forward pass should succeed in training mode"
        );
    }

    #[test]
    fn dense_forward_inference_mode() {
        let mut layer = Dense::new(5, 3, None, None);
        let input = Tensor::zeros(vec![2, 5]);

        let result = layer.forward(&input, false);

        assert!(
            result.is_ok(),
            "Forward pass should succeed in inference mode"
        );
    }

    #[test]
    fn dense_backward_requires_forward_first() {
        let mut layer = Dense::new(4, 3, None, None);
        let grad_output = Tensor::zeros(vec![2, 3]);

        let result = layer.backward(&grad_output);

        assert!(
            result.is_err(),
            "Backward should fail if forward not called first"
        );
    }

    #[test]
    fn dense_backward_correct_input_gradient_shape() {
        let mut layer = Dense::new(4, 3, None, None);
        let input = Tensor::random(vec![2, 4], -1.0..1.0);

        layer.forward(&input, true).unwrap();

        let grad_output = Tensor::new(vec![2, 3], vec![1.0; 6]).unwrap();
        let grad_input = layer.backward(&grad_output).unwrap();

        assert_eq!(grad_input.dimension, 2);
        assert_eq!(grad_input.shape, vec![2, 4]);
    }

    #[test]
    fn dense_backward_updates_weight_gradients() {
        let mut layer = Dense::new(3, 2, None, None);
        let input = Tensor::new(vec![1, 3], vec![1.0; 3]).unwrap();
        let grad_output = Tensor::new(vec![1, 2], vec![1.0; 2]).unwrap();

        layer.forward(&input, true).unwrap();
        layer.backward(&grad_output).unwrap();

        let params = layer.get_parameters().unwrap();
        let w_grads = params.w_grads.lock().unwrap();

        assert!(
            w_grads.data.iter().any(|&x| x != 0.0),
            "Weight gradients should be non-zero after backward"
        );
    }

    #[test]
    fn dense_backward_updates_bias_gradients() {
        let mut layer = Dense::new(3, 2, None, None);
        let input = Tensor::new(vec![1, 3], vec![1.0; 3]).unwrap();
        let grad_output = Tensor::new(vec![1, 2], vec![1.0; 2]).unwrap();

        layer.forward(&input, true).unwrap();
        layer.backward(&grad_output).unwrap();

        let params = layer.get_parameters().unwrap();
        let b_grads = params.b_grads.lock().unwrap();

        assert!(
            b_grads.data.iter().all(|&x| x == 1.0),
            "Bias gradients should equal sum of grad_output (all 1s here)"
        );
    }

    #[test]
    fn dense_parameters_shared_across_calls() {
        let layer = Dense::new(2, 2, None, None);

        let params1 = layer.get_parameters().unwrap();
        let params2 = layer.get_parameters().unwrap();

        let w1_ptr = Arc::as_ptr(&params1.weights);
        let w2_ptr = Arc::as_ptr(&params2.weights);

        assert_eq!(
            w1_ptr, w2_ptr,
            "get_parameters() should return shared references (same Arc)"
        );
    }

    #[test]
    fn dense_he_initialization() {
        let layer = Dense::new(100, 50, Some(Init::He), None);
        let params = layer.get_parameters().unwrap();
        let weights = params.weights.lock().unwrap();

        let variance = weights.data.iter().map(|&x| x * x).sum::<f32>() / weights.data.len() as f32;

        let expected_variance = 2.0 / 100.0;

        assert!(
            (variance - expected_variance).abs() < 0.01,
            "He init variance should be ~2/fan_in (expected {}, got {})",
            expected_variance,
            variance
        );
    }

    #[test]
    fn dense_xavier_initialization() {
        let layer = Dense::new(100, 50, Some(Init::Xavier), None);
        let params = layer.get_parameters().unwrap();
        let weights = params.weights.lock().unwrap();

        let variance = weights.data.iter().map(|&x| x * x).sum::<f32>() / weights.data.len() as f32;

        let expected_variance = 2.0 / (100.0 + 50.0);

        assert!(
            (variance - expected_variance).abs() < 0.01,
            "Xavier init variance should be ~2/(fan_in+fan_out) (expected {}, got {})",
            expected_variance,
            variance
        );
    }

    #[test]
    fn dense_biases_initialized_to_zero() {
        let layer = Dense::new(10, 5, None, None);
        let params = layer.get_parameters().unwrap();
        let biases = params.biases.lock().unwrap();

        assert!(
            biases.data.iter().all(|&x| x == 0.0),
            "Biases should always be initialized to zero"
        );
    }

    #[test]
    fn dense_multiple_forward_backward_cycles() {
        let mut layer = Dense::new(3, 2, None, None);

        for _ in 0..5 {
            let input = Tensor::random(vec![2, 3], -1.0..1.0);
            let _output = layer.forward(&input, true).unwrap();

            let grad_output = Tensor::new(vec![2, 2], vec![1.0; 4]).unwrap();
            let grad_input = layer.backward(&grad_output);

            assert!(grad_input.is_ok(), "Should handle multiple cycles");
        }
    }

    #[test]
    fn dense_batch_processing() {
        let mut layer = Dense::new(4, 3, None, None);

        let batch_sizes = vec![1, 8, 32, 128];

        for batch_size in batch_sizes {
            let input = Tensor::random(vec![batch_size, 4], -1.0..1.0);
            let output = layer.forward(&input, true).unwrap();

            assert_eq!(output.shape, vec![batch_size, 3]);

            let grad_output = Tensor::new(vec![batch_size, 3], vec![1.0; batch_size * 3]).unwrap();
            let grad_input = layer.backward(&grad_output).unwrap();

            assert_eq!(grad_input.shape, vec![batch_size, 4]);
        }
    }
}

#[cfg(test)]
mod softmax_tests {
    use super::*;

    #[test]
    fn softmax_backward_standalone() {
        let input = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        let mut softmax = Softmax::new();
        let output = softmax.forward(&input, true).unwrap();

        let sum: f32 = output.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        let grad_output = Tensor::new(vec![3], vec![0.5, -0.3, 0.1]).unwrap();
        let grad_input = softmax.backward(&grad_output).unwrap();

        let sigma = output.data.as_slice().unwrap();
        let d_l_dsigma = grad_output.data.as_slice().unwrap();

        let weighted: Vec<f32> = sigma
            .iter()
            .zip(d_l_dsigma.iter())
            .map(|(s, g)| s * g)
            .collect();
        let sum_weighted: f32 = weighted.iter().sum();

        let expected: Vec<f32> = sigma
            .iter()
            .zip(d_l_dsigma.iter())
            .map(|(s, g)| s * (g - sum_weighted))
            .collect();

        let computed = grad_input.data.as_slice().unwrap();
        for (c, e) in computed.iter().zip(expected.iter()) {
            assert!((c - e).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_backward_with_uniform_gradient() {
        let input = Tensor::new(vec![3], vec![0.5, 1.0, 1.5]).unwrap();

        let mut softmax = Softmax::new();
        let _output = softmax.forward(&input, true).unwrap();

        let grad_output = Tensor::new(vec![3], vec![1.0, 1.0, 1.0]).unwrap();
        let grad_input = softmax.backward(&grad_output).unwrap();

        let computed = grad_input.data.as_slice().unwrap();
        let sum: f32 = computed.iter().sum();

        assert!(sum.abs() < 1e-6);
    }

    #[test]
    fn softmax_backward_2d_batch() {
        let input = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let mut softmax = Softmax::new();
        let output = softmax.forward(&input, true).unwrap();

        let row0_sum: f32 = output.data.iter().take(3).sum();
        let row1_sum: f32 = output.data.iter().skip(3).sum();
        assert!((row0_sum - 1.0).abs() < 1e-6);
        assert!((row1_sum - 1.0).abs() < 1e-6);

        let grad_output = Tensor::new(vec![2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let grad_input = softmax.backward(&grad_output).unwrap();

        assert_eq!(grad_input.shape, vec![2, 3]);
    }
}
