//! Test suite for parallel matrix multiplication
//!
//! Tests verify correctness and consistency of par_dot against sequential dot product.

use crate::math::matmul::par_dot;
use crate::tensor::Tensor;
use ndarray::Array2;

#[cfg(test)]
mod par_dot_tests {
    use super::*;

    #[test]
    fn par_dot_2x2_matrices() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let result = par_dot(a.view(), b.view());

        assert_eq!(result[[0, 0]], 19.0);
        assert_eq!(result[[0, 1]], 22.0);
        assert_eq!(result[[1, 0]], 43.0);
        assert_eq!(result[[1, 1]], 50.0);
    }

    #[test]
    fn par_dot_identity_matrix() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let identity = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        let result = par_dot(a.view(), identity.view());

        assert_eq!(result.as_slice().unwrap(), a.as_slice().unwrap());
    }

    #[test]
    fn par_dot_zero_matrix() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let zeros = Array2::zeros((2, 2));

        let result = par_dot(a.view(), zeros.view());

        assert_eq!(result.as_slice().unwrap(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn par_dot_rectangular_matrices() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let result = par_dot(a.view(), b.view());

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 58.0);
        assert_eq!(result[[0, 1]], 64.0);
        assert_eq!(result[[1, 0]], 139.0);
        assert_eq!(result[[1, 1]], 154.0);
    }

    #[test]
    fn par_dot_single_row_matrix() {
        let a = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

        let result = par_dot(a.view(), b.view());

        assert_eq!(result.shape(), &[1, 2]);
        assert_eq!(result[[0, 0]], 40.0);
        assert_eq!(result[[0, 1]], 46.0);
    }

    #[test]
    fn par_dot_single_column_matrix() {
        let a = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array2::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).unwrap();

        let result = par_dot(a.view(), b.view());

        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(result[[0, 0]], 4.0);
        assert_eq!(result[[1, 1]], 10.0);
        assert_eq!(result[[2, 2]], 18.0);
    }

    #[test]
    fn par_dot_large_matrix() {
        let a = Array2::from_shape_fn((100, 100), |(i, j)| (i + j) as f32);
        let b = Array2::from_shape_fn((100, 100), |(i, j)| (i * j) as f32);

        let result = par_dot(a.view(), b.view());

        assert_eq!(result.shape(), &[100, 100]);
        assert!(result.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn par_dot_versus_sequential() {
        let a = Array2::from_shape_vec((3, 4), (0..12).map(|x| x as f32).collect()).unwrap();
        let b = Array2::from_shape_vec((4, 3), (0..12).map(|x| x as f32).collect()).unwrap();

        let par_result = par_dot(a.view(), b.view());
        let seq_result = a.dot(&b);

        assert_eq!(par_result.shape(), seq_result.shape());
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (par_result[[i, j]] - seq_result[[i, j]]).abs() < 1e-5,
                    "Mismatch at [{}, {}]: par={}, seq={}",
                    i,
                    j,
                    par_result[[i, j]],
                    seq_result[[i, j]]
                );
            }
        }
    }

    #[test]
    fn par_dot_negative_values() {
        let a = Array2::from_shape_vec((2, 2), vec![-1.0, -2.0, -3.0, -4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = par_dot(a.view(), b.view());

        assert_eq!(result[[0, 0]], -7.0);
        assert_eq!(result[[0, 1]], -10.0);
        assert_eq!(result[[1, 0]], -15.0);
        assert_eq!(result[[1, 1]], -22.0);
    }

    #[test]
    fn par_dot_mixed_values() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, -2.0, 3.0, -4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![-1.0, 2.0, -3.0, 4.0]).unwrap();

        let result = par_dot(a.view(), b.view());

        assert_eq!(result[[0, 0]], 5.0);
        assert_eq!(result[[0, 1]], -6.0);
        assert_eq!(result[[1, 0]], 9.0);
        assert_eq!(result[[1, 1]], -10.0);
    }

    #[test]
    fn par_dot_with_floats() {
        let a = Array2::from_shape_vec((2, 2), vec![1.5, 2.5, 3.5, 4.5]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![0.5, 1.5, 2.5, 3.5]).unwrap();

        let result = par_dot(a.view(), b.view());

        assert!((result[[0, 0]] - 7.0).abs() < 1e-5);
        assert!((result[[0, 1]] - 11.0).abs() < 1e-5);
        assert!((result[[1, 0]] - 13.0).abs() < 1e-5);
        assert!((result[[1, 1]] - 21.0).abs() < 1e-5);
    }

    #[test]
    fn par_dot_consistency_across_runs() {
        let a = Tensor::random(vec![50, 50], 0.0..100.0);
        let b = Tensor::random(vec![50, 50], 0.0..100.0);

        let a_view = a.data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_view = b.data.view().into_dimensionality::<ndarray::Ix2>().unwrap();

        let result1 = par_dot(a_view, b_view);
        let result2 = par_dot(a_view, b_view);

        assert_eq!(result1.as_slice().unwrap(), result2.as_slice().unwrap());
    }

    #[test]
    fn par_dot_very_small_values() {
        let a = Array2::from_shape_vec((2, 2), vec![1e-10, 1e-20, 1e-15, 1e-25]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1e10, 1e20, 1e15, 1e25]).unwrap();

        let result = par_dot(a.view(), b.view());

        assert!(result.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn par_dot_output_shape() {
        let test_cases = vec![(2, 3, 4), (5, 7, 3), (10, 1, 10), (1, 100, 1)];

        for (m, n, p) in test_cases {
            let a = Array2::zeros((m, n));
            let b = Array2::zeros((n, p));

            let result = par_dot(a.view(), b.view());

            assert_eq!(
                result.shape(),
                &[m, p],
                "Shape mismatch for ({}, {}) × ({}, {})",
                m,
                n,
                n,
                p
            );
        }
    }
}
