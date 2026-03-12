//! Test suite for the optimizers module.
//!
//! Tests verify the `GradientDescent` optimizer with new pattern:
//! - register_params() for parameter registration
//! - step() for weight updates
//! - zero_grad() for gradient reset
//! - Correctness with Arc<Mutex<>> shared ownership
//! - Parallelization with Rayon

use std::sync::{Arc, Mutex};

use crate::optimizers::{gradient_descent::GradientDescent, Optimizer, ParametersRef};
use crate::tensor::Tensor;

const EPSILON: f32 = 1e-5;

// ─────────────────────────────────────────────────────────────
//  Helpers: build tensors from flat slices
// ─────────────────────────────────────────────────────────────
fn tensor1d(data: &[f32]) -> Tensor {
    Tensor::new(vec![data.len()], data.to_vec()).unwrap()
}

fn tensor2d(rows: usize, cols: usize, data: &[f32]) -> Tensor {
    Tensor::new(vec![rows, cols], data.to_vec()).unwrap()
}

fn approx_equal(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

// ─────────────────────────────────────────────────────────────
//  Test: register_params stores parameters correctly
// ─────────────────────────────────────────────────────────────
#[test]
fn register_params_stores_references() {
    let mut optimizer = GradientDescent {
        learning_rate: 0.01,
        params: vec![],
    };

    let weights = Arc::new(Mutex::new(tensor1d(&[1.0, 2.0, 3.0])));
    let biases = Arc::new(Mutex::new(tensor1d(&[0.5])));
    let w_grads = Arc::new(Mutex::new(tensor1d(&[0.1, 0.2, 0.3])));
    let b_grads = Arc::new(Mutex::new(tensor1d(&[0.05])));

    let param_ref = ParametersRef {
        weights: weights.clone(),
        biases: biases.clone(),
        w_grads: w_grads.clone(),
        b_grads: b_grads.clone(),
    };

    optimizer.register_params(vec![param_ref]);

    assert_eq!(optimizer.params.len(), 1);
}

// ─────────────────────────────────────────────────────────────
//  Test: step() updates weights correctly (single layer)
// ─────────────────────────────────────────────────────────────
#[test]
fn step_updates_weights_single_layer() {
    let mut optimizer = GradientDescent {
        learning_rate: 0.1,
        params: vec![],
    };

    let weights = Arc::new(Mutex::new(tensor1d(&[1.0, 2.0, 3.0])));
    let biases = Arc::new(Mutex::new(tensor1d(&[0.5])));
    let w_grads = Arc::new(Mutex::new(tensor1d(&[0.1, 0.2, 0.3])));
    let b_grads = Arc::new(Mutex::new(tensor1d(&[0.05])));

    optimizer.register_params(vec![ParametersRef {
        weights: weights.clone(),
        biases: biases.clone(),
        w_grads: w_grads.clone(),
        b_grads: b_grads.clone(),
    }]);

    optimizer.step().unwrap();

    let w_data = weights.lock().unwrap().data.clone();
    let b_data = biases.lock().unwrap().data.clone();

    assert!(approx_equal(w_data[0], 1.0 - 0.1 * 0.1));
    assert!(approx_equal(w_data[1], 2.0 - 0.1 * 0.2));
    assert!(approx_equal(w_data[2], 3.0 - 0.1 * 0.3));
    assert!(approx_equal(b_data[0], 0.5 - 0.1 * 0.05));
}

// ─────────────────────────────────────────────────────────────
//  Test: step() updates multiple layers independently
// ─────────────────────────────────────────────────────────────
#[test]
fn step_updates_multiple_layers() {
    let mut optimizer = GradientDescent {
        learning_rate: 0.1,
        params: vec![],
    };

    let w1 = Arc::new(Mutex::new(tensor1d(&[1.0, 2.0])));
    let b1 = Arc::new(Mutex::new(tensor1d(&[0.5])));
    let w1_grads = Arc::new(Mutex::new(tensor1d(&[0.1, 0.2])));
    let b1_grads = Arc::new(Mutex::new(tensor1d(&[0.05])));

    let w2 = Arc::new(Mutex::new(tensor1d(&[3.0, 4.0])));
    let b2 = Arc::new(Mutex::new(tensor1d(&[1.0])));
    let w2_grads = Arc::new(Mutex::new(tensor1d(&[0.3, 0.4])));
    let b2_grads = Arc::new(Mutex::new(tensor1d(&[0.1])));

    optimizer.register_params(vec![
        ParametersRef {
            weights: w1.clone(),
            biases: b1.clone(),
            w_grads: w1_grads.clone(),
            b_grads: b1_grads.clone(),
        },
        ParametersRef {
            weights: w2.clone(),
            biases: b2.clone(),
            w_grads: w2_grads.clone(),
            b_grads: b2_grads.clone(),
        },
    ]);

    optimizer.step().unwrap();

    let w1_data = w1.lock().unwrap().data.clone();
    let w2_data = w2.lock().unwrap().data.clone();

    assert!(approx_equal(w1_data[0], 1.0 - 0.1 * 0.1));
    assert!(approx_equal(w1_data[1], 2.0 - 0.1 * 0.2));
    assert!(approx_equal(w2_data[0], 3.0 - 0.1 * 0.3));
    assert!(approx_equal(w2_data[1], 4.0 - 0.1 * 0.4));
}

// ─────────────────────────────────────────────────────────────
//  Test: zero_grad() resets gradients to zero
// ─────────────────────────────────────────────────────────────
#[test]
fn zero_grad_resets_gradients() {
    let mut optimizer = GradientDescent {
        learning_rate: 0.1,
        params: vec![],
    };

    let weights = Arc::new(Mutex::new(tensor1d(&[1.0, 2.0])));
    let biases = Arc::new(Mutex::new(tensor1d(&[0.5])));
    let w_grads = Arc::new(Mutex::new(tensor1d(&[0.1, 0.2])));
    let b_grads = Arc::new(Mutex::new(tensor1d(&[0.05])));

    optimizer.register_params(vec![ParametersRef {
        weights: weights.clone(),
        biases: biases.clone(),
        w_grads: w_grads.clone(),
        b_grads: b_grads.clone(),
    }]);

    optimizer.zero_grad().unwrap();

    let w_grad_data = w_grads.lock().unwrap().data.clone();
    let b_grad_data = b_grads.lock().unwrap().data.clone();

    assert!(approx_equal(w_grad_data[0], 0.0));
    assert!(approx_equal(w_grad_data[1], 0.0));
    assert!(approx_equal(b_grad_data[0], 0.0));
}

// ─────────────────────────────────────────────────────────────
//  Test: gradient accumulation without zero_grad
// ─────────────────────────────────────────────────────────────
#[test]
fn gradients_accumulate_without_zero_grad() {
    let mut optimizer = GradientDescent {
        learning_rate: 0.1,
        params: vec![],
    };

    let weights = Arc::new(Mutex::new(tensor1d(&[1.0])));
    let biases = Arc::new(Mutex::new(tensor1d(&[0.5])));
    let w_grads = Arc::new(Mutex::new(tensor1d(&[0.1])));
    let b_grads = Arc::new(Mutex::new(tensor1d(&[0.05])));

    optimizer.register_params(vec![ParametersRef {
        weights: weights.clone(),
        biases: biases.clone(),
        w_grads: w_grads.clone(),
        b_grads: b_grads.clone(),
    }]);

    {
        let mut w_grad = w_grads.lock().unwrap();
        *w_grad = (&*w_grad + &tensor1d(&[0.1])).unwrap();
    }

    let w_grad_data = w_grads.lock().unwrap().data.clone();
    assert!(approx_equal(w_grad_data[0], 0.2));
}

// ─────────────────────────────────────────────────────────────
//  Test: multiple step() calls converge weights
// ─────────────────────────────────────────────────────────────
#[test]
fn multiple_steps_converge() {
    let mut optimizer = GradientDescent {
        learning_rate: 0.1,
        params: vec![],
    };

    let weights = Arc::new(Mutex::new(tensor1d(&[10.0])));
    let biases = Arc::new(Mutex::new(tensor1d(&[5.0])));
    let w_grads = Arc::new(Mutex::new(tensor1d(&[1.0])));
    let b_grads = Arc::new(Mutex::new(tensor1d(&[0.5])));

    optimizer.register_params(vec![ParametersRef {
        weights: weights.clone(),
        biases: biases.clone(),
        w_grads: w_grads.clone(),
        b_grads: b_grads.clone(),
    }]);

    for _ in 0..10 {
        optimizer.step().unwrap();
    }

    let w_data = weights.lock().unwrap().data.clone();
    let b_data = biases.lock().unwrap().data.clone();

    assert!(approx_equal(w_data[0], 10.0 - 10.0 * 0.1 * 1.0));
    assert!(approx_equal(b_data[0], 5.0 - 10.0 * 0.1 * 0.5));
}

// ─────────────────────────────────────────────────────────────
//  Test: learning rate affects update magnitude
// ─────────────────────────────────────────────────────────────
#[test]
fn learning_rate_affects_update() {
    let weights1 = Arc::new(Mutex::new(tensor1d(&[1.0])));
    let w_grads1 = Arc::new(Mutex::new(tensor1d(&[0.1])));
    let biases1 = Arc::new(Mutex::new(tensor1d(&[0.5])));
    let b_grads1 = Arc::new(Mutex::new(tensor1d(&[0.05])));

    let weights2 = Arc::new(Mutex::new(tensor1d(&[1.0])));
    let w_grads2 = Arc::new(Mutex::new(tensor1d(&[0.1])));
    let biases2 = Arc::new(Mutex::new(tensor1d(&[0.5])));
    let b_grads2 = Arc::new(Mutex::new(tensor1d(&[0.05])));

    let mut opt1 = GradientDescent {
        learning_rate: 0.01,
        params: vec![],
    };
    opt1.register_params(vec![ParametersRef {
        weights: weights1.clone(),
        biases: biases1.clone(),
        w_grads: w_grads1,
        b_grads: b_grads1,
    }]);

    let mut opt2 = GradientDescent {
        learning_rate: 0.1,
        params: vec![],
    };
    opt2.register_params(vec![ParametersRef {
        weights: weights2.clone(),
        biases: biases2.clone(),
        w_grads: w_grads2,
        b_grads: b_grads2,
    }]);

    opt1.step().unwrap();
    opt2.step().unwrap();

    let w1 = weights1.lock().unwrap().data[0];
    let w2 = weights2.lock().unwrap().data[0];

    assert!(w1 > w2);
}

// ─────────────────────────────────────────────────────────────
//  Test: step() preserves tensor shapes
// ─────────────────────────────────────────────────────────────
#[test]
fn step_preserves_shapes() {
    let mut optimizer = GradientDescent {
        learning_rate: 0.1,
        params: vec![],
    };

    let weights = Arc::new(Mutex::new(tensor2d(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));
    let biases = Arc::new(Mutex::new(tensor1d(&[0.5, 1.0])));
    let w_grads = Arc::new(Mutex::new(tensor2d(2, 3, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])));
    let b_grads = Arc::new(Mutex::new(tensor1d(&[0.05, 0.1])));

    optimizer.register_params(vec![ParametersRef {
        weights: weights.clone(),
        biases: biases.clone(),
        w_grads: w_grads.clone(),
        b_grads: b_grads.clone(),
    }]);

    optimizer.step().unwrap();

    let w_shape = weights.lock().unwrap().shape.clone();
    let b_shape = biases.lock().unwrap().shape.clone();

    assert_eq!(w_shape, vec![2, 3]);
    assert_eq!(b_shape, vec![2]);
}

// ─────────────────────────────────────────────────────────────
//  Test: zero_grad() after step() allows fresh backward
// ─────────────────────────────────────────────────────────────
#[test]
fn zero_grad_enables_fresh_backward() {
    let mut optimizer = GradientDescent {
        learning_rate: 0.1,
        params: vec![],
    };

    let weights = Arc::new(Mutex::new(tensor1d(&[1.0])));
    let biases = Arc::new(Mutex::new(tensor1d(&[0.5])));
    let w_grads = Arc::new(Mutex::new(tensor1d(&[0.1])));
    let b_grads = Arc::new(Mutex::new(tensor1d(&[0.05])));

    optimizer.register_params(vec![ParametersRef {
        weights: weights.clone(),
        biases: biases.clone(),
        w_grads: w_grads.clone(),
        b_grads: b_grads.clone(),
    }]);

    optimizer.step().unwrap();
    let w1 = weights.lock().unwrap().data[0];

    optimizer.zero_grad().unwrap();

    {
        let mut w_grad = w_grads.lock().unwrap();
        *w_grad = tensor1d(&[0.2]);
    }

    optimizer.step().unwrap();
    let w2 = weights.lock().unwrap().data[0];

    assert!(approx_equal(w2, w1 - 0.1 * 0.2));
}

// ─────────────────────────────────────────────────────────────
//  Test: shared ownership - layer and optimizer see same memory
// ─────────────────────────────────────────────────────────────
#[test]
fn shared_ownership_works() {
    let weights = Arc::new(Mutex::new(tensor1d(&[1.0, 2.0])));

    let layer_weights = weights.clone();
    let optimizer_weights = weights.clone();

    {
        let mut w = layer_weights.lock().unwrap();
        *w = tensor1d(&[10.0, 20.0]);
    }

    let opt_data = optimizer_weights.lock().unwrap().data.clone();
    assert!(approx_equal(opt_data[0], 10.0));
    assert!(approx_equal(opt_data[1], 20.0));
}

// ─────────────────────────────────────────────────────────────
//  Test: zero learning rate means no update
// ─────────────────────────────────────────────────────────────
#[test]
fn zero_learning_rate_no_update() {
    let mut optimizer = GradientDescent {
        learning_rate: 0.0,
        params: vec![],
    };

    let weights = Arc::new(Mutex::new(tensor1d(&[1.0, 2.0])));
    let biases = Arc::new(Mutex::new(tensor1d(&[0.5])));
    let w_grads = Arc::new(Mutex::new(tensor1d(&[0.1, 0.2])));
    let b_grads = Arc::new(Mutex::new(tensor1d(&[0.05])));

    optimizer.register_params(vec![ParametersRef {
        weights: weights.clone(),
        biases: biases.clone(),
        w_grads: w_grads.clone(),
        b_grads: b_grads.clone(),
    }]);

    optimizer.step().unwrap();

    let w_data = weights.lock().unwrap().data.clone();
    assert!(approx_equal(w_data[0], 1.0));
    assert!(approx_equal(w_data[1], 2.0));
}

// ─────────────────────────────────────────────────────────────
//  Test: negative gradients increase weights
// ─────────────────────────────────────────────────────────────
#[test]
fn negative_gradients_increase_weights() {
    let mut optimizer = GradientDescent {
        learning_rate: 0.1,
        params: vec![],
    };

    let weights = Arc::new(Mutex::new(tensor1d(&[1.0])));
    let biases = Arc::new(Mutex::new(tensor1d(&[0.5])));
    let w_grads = Arc::new(Mutex::new(tensor1d(&[-0.5])));
    let b_grads = Arc::new(Mutex::new(tensor1d(&[-0.1])));

    optimizer.register_params(vec![ParametersRef {
        weights: weights.clone(),
        biases: biases.clone(),
        w_grads: w_grads.clone(),
        b_grads: b_grads.clone(),
    }]);

    optimizer.step().unwrap();

    let w_data = weights.lock().unwrap().data[0];
    let b_data = biases.lock().unwrap().data[0];

    assert!(approx_equal(w_data, 1.0 - 0.1 * (-0.5)));
    assert!(approx_equal(b_data, 0.5 - 0.1 * (-0.1)));
    assert!(w_data > 1.0);
    assert!(b_data > 0.5);
}
