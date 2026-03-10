//! Test suite for the optimizers module.
//!
//! Tests verify the `GradientDescent` optimizer: correctness of weight/bias updates,
//! effect of learning rate, convergence behaviour over multiple steps,
//! and shape/dimension preservation.

use crate::optimizers::{gradient_descent::GradientDescent, Optimizer};
use crate::tensor::Tensor;

const EPSILON: f32 = 1e-5;

// ─────────────────────────────────────────────────────────────
//  Helper: build a 1-D tensor from a flat slice
// ─────────────────────────────────────────────────────────────
fn tensor1d(data: &[f32]) -> Tensor {
    Tensor::new(vec![data.len()], data.to_vec()).unwrap()
}

fn tensor2d(rows: usize, cols: usize, data: &[f32]) -> Tensor {
    Tensor::new(vec![rows, cols], data.to_vec()).unwrap()
}

// ─────────────────────────────────────────────────────────────
#[cfg(test)]
mod gradient_descent_tests {
    use super::*;

    // ── basic update correctness ──────────────────────────────

    /// w_new = w - lr * grad_w   (scalar case, easy to verify by hand)
    #[test]
    fn update_weights_simple() {
        let mut gd = GradientDescent { learning_rate: 0.1 };

        let mut weights = tensor1d(&[1.0, 2.0, 3.0]);
        let mut biases = tensor1d(&[0.0]);
        let w_grads = tensor1d(&[10.0, 20.0, 30.0]);
        let b_grads = tensor1d(&[0.0]);

        gd.update(&mut weights, &mut biases, &w_grads, &b_grads, 0)
            .unwrap();

        // w_new = [1 - 0.1*10, 2 - 0.1*20, 3 - 0.1*30] = [0, 0, 0]
        for &v in weights.data.iter() {
            assert!(v.abs() < EPSILON, "expected 0.0, got {v}");
        }
    }

    #[test]
    fn update_biases_simple() {
        let mut gd = GradientDescent { learning_rate: 0.5 };

        let mut weights = tensor1d(&[0.0]);
        let mut biases = tensor1d(&[4.0, 6.0]);
        let w_grads = tensor1d(&[0.0]);
        let b_grads = tensor1d(&[2.0, 4.0]);

        gd.update(&mut weights, &mut biases, &w_grads, &b_grads, 0)
            .unwrap();

        // b_new = [4 - 0.5*2, 6 - 0.5*4] = [3, 4]
        assert!((biases.data[0] - 3.0).abs() < EPSILON);
        assert!((biases.data[1] - 4.0).abs() < EPSILON);
    }

    // ── learning rate scaling ─────────────────────────────────

    /// A larger lr should produce a larger weight change.
    #[test]
    fn larger_lr_produces_larger_update() {
        let grads = tensor1d(&[1.0, 1.0]);

        let mut w_small = tensor1d(&[5.0, 5.0]);
        let mut b_dummy = tensor1d(&[0.0]);
        let b_grads = tensor1d(&[0.0]);

        let mut gd_small = GradientDescent { learning_rate: 0.01 };
        gd_small
            .update(&mut w_small, &mut b_dummy, &grads, &b_grads, 0)
            .unwrap();

        let mut w_large = tensor1d(&[5.0, 5.0]);
        let mut b_dummy2 = tensor1d(&[0.0]);

        let mut gd_large = GradientDescent { learning_rate: 1.0 };
        gd_large
            .update(&mut w_large, &mut b_dummy2, &grads, &b_grads, 0)
            .unwrap();

        // |w_large - 5| > |w_small - 5|
        let delta_small = (w_small.data[0] - 5.0).abs();
        let delta_large = (w_large.data[0] - 5.0).abs();
        assert!(delta_large > delta_small);
    }

    /// lr = 0 → no update at all.
    #[test]
    fn zero_lr_no_update() {
        let mut gd = GradientDescent { learning_rate: 0.0 };

        let mut weights = tensor1d(&[3.0, -1.0]);
        let mut biases = tensor1d(&[2.0]);
        let w_grads = tensor1d(&[100.0, 200.0]);
        let b_grads = tensor1d(&[50.0]);

        gd.update(&mut weights, &mut biases, &w_grads, &b_grads, 0)
            .unwrap();

        assert!((weights.data[0] - 3.0).abs() < EPSILON);
        assert!((weights.data[1] - (-1.0)).abs() < EPSILON);
        assert!((biases.data[0] - 2.0).abs() < EPSILON);
    }

    // ── zero gradients ────────────────────────────────────────

    /// Zero gradients → parameters must not change.
    #[test]
    fn zero_gradients_no_change() {
        let mut gd = GradientDescent { learning_rate: 0.5 };

        let mut weights = tensor1d(&[1.0, 2.0, 3.0]);
        let mut biases = tensor1d(&[4.0, 5.0]);
        let w_grads = tensor1d(&[0.0, 0.0, 0.0]);
        let b_grads = tensor1d(&[0.0, 0.0]);

        gd.update(&mut weights, &mut biases, &w_grads, &b_grads, 0)
            .unwrap();

        assert!((weights.data[0] - 1.0).abs() < EPSILON);
        assert!((weights.data[1] - 2.0).abs() < EPSILON);
        assert!((weights.data[2] - 3.0).abs() < EPSILON);
        assert!((biases.data[0] - 4.0).abs() < EPSILON);
        assert!((biases.data[1] - 5.0).abs() < EPSILON);
    }

    // ── negative gradients ────────────────────────────────────

    /// Negative gradients → parameters increase.
    #[test]
    fn negative_gradients_increase_params() {
        let mut gd = GradientDescent { learning_rate: 0.1 };

        let mut weights = tensor1d(&[1.0]);
        let mut biases = tensor1d(&[1.0]);
        let w_grads = tensor1d(&[-5.0]);
        let b_grads = tensor1d(&[-5.0]);

        gd.update(&mut weights, &mut biases, &w_grads, &b_grads, 0)
            .unwrap();

        assert!(weights.data[0] > 1.0);
        assert!(biases.data[0] > 1.0);
    }

    // ── shape preservation ────────────────────────────────────

    #[test]
    fn update_preserves_weight_shape() {
        let mut gd = GradientDescent { learning_rate: 0.01 };

        let mut weights = tensor2d(3, 4, &[1.0; 12]);
        let mut biases = tensor1d(&[0.0; 4]);
        let w_grads = tensor2d(3, 4, &[0.1; 12]);
        let b_grads = tensor1d(&[0.1; 4]);

        gd.update(&mut weights, &mut biases, &w_grads, &b_grads, 0)
            .unwrap();

        assert_eq!(weights.shape, vec![3, 4]);
        assert_eq!(weights.dimension, 2);
    }

    #[test]
    fn update_preserves_bias_shape() {
        let mut gd = GradientDescent { learning_rate: 0.01 };

        let mut weights = tensor2d(2, 3, &[0.0; 6]);
        let mut biases = tensor1d(&[1.0, 2.0, 3.0]);
        let w_grads = tensor2d(2, 3, &[0.0; 6]);
        let b_grads = tensor1d(&[0.1, 0.2, 0.3]);

        gd.update(&mut weights, &mut biases, &w_grads, &b_grads, 0)
            .unwrap();

        assert_eq!(biases.shape, vec![3]);
        assert_eq!(biases.dimension, 1);
    }

    // ── convergence over multiple steps ───────────────────────

    /// Constant gradient → weight should converge monotonically toward 0.
    #[test]
    fn multiple_steps_reduces_weight() {
        let mut gd = GradientDescent { learning_rate: 0.1 };

        let mut weights = tensor1d(&[10.0]);
        let mut biases = tensor1d(&[0.0]);
        let b_grads = tensor1d(&[0.0]);

        for step in 0..50 {
            // gradient is simply the current weight value (gradient descent on 0.5*w^2)
            let w_val = weights.data[0];
            let w_grads = tensor1d(&[w_val]);
            gd.update(&mut weights, &mut biases, &w_grads, &b_grads, step)
                .unwrap();
        }

        // After 50 steps with lr=0.1, w_new = w * (1 - 0.1)^50 ≈ 10 * 0.005 = 0.05
        assert!(weights.data[0].abs() < 1.0, "weight did not converge: {}", weights.data[0]);
    }

    /// Update with the `step` argument ignored (GradientDescent is stateless).
    #[test]
    fn step_argument_has_no_effect() {
        // GradientDescent doesn't use `step`, so update at step 0 == step 1000
        let grads = tensor1d(&[1.0, 1.0]);
        let b_grads = tensor1d(&[0.0]);

        let mut gd0 = GradientDescent { learning_rate: 0.3 };
        let mut w0 = tensor1d(&[5.0, 5.0]);
        let mut b0 = tensor1d(&[0.0]);
        gd0.update(&mut w0, &mut b0, &grads, &b_grads, 0).unwrap();

        let mut gd1000 = GradientDescent { learning_rate: 0.3 };
        let mut w1000 = tensor1d(&[5.0, 5.0]);
        let mut b1000 = tensor1d(&[0.0]);
        gd1000
            .update(&mut w1000, &mut b1000, &grads, &b_grads, 1000)
            .unwrap();

        for (a, b) in w0.data.iter().zip(w1000.data.iter()) {
            assert!((a - b).abs() < EPSILON);
        }
    }

    // ── 2-D parameters (realistic layer scenario) ────────────

    #[test]
    fn update_2d_weights_correct_values() {
        let mut gd = GradientDescent { learning_rate: 1.0 };

        // weights: [[2, 4], [6, 8]]
        let mut weights = tensor2d(2, 2, &[2.0, 4.0, 6.0, 8.0]);
        // biases: [1, 2]
        let mut biases = tensor1d(&[1.0, 2.0]);
        // gradients: [[1, 1], [1, 1]]
        let w_grads = tensor2d(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let b_grads = tensor1d(&[1.0, 1.0]);

        gd.update(&mut weights, &mut biases, &w_grads, &b_grads, 0)
            .unwrap();

        // w_new = [[1,3],[5,7]], b_new = [0, 1]  (lr=1.0)
        let expected_w = [1.0, 3.0, 5.0, 7.0];
        for (got, &exp) in weights.data.iter().zip(expected_w.iter()) {
            assert!((got - exp).abs() < EPSILON, "expected {exp}, got {got}");
        }
        assert!((biases.data[0] - 0.0).abs() < EPSILON);
        assert!((biases.data[1] - 1.0).abs() < EPSILON);
    }
}
