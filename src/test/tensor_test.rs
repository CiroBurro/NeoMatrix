//! Comprehensive test suite for Tensor operations
//!
//! Tests cover construction, arithmetic, linear algebra, shape operations,
//! concatenation, serialization, and iteration using PyO3 0.26.x patterns.

use crate::structures::tensor::Tensor;
use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::py_run;
use pyo3::types::PyDict;

// =============================================================================
// TENSOR CONSTRUCTION
// =============================================================================

#[cfg(test)]
mod tensor_construction {
    use super::*;

    #[test]
    fn new_creates_tensor_with_correct_shape_and_content() {
        let shape = vec![2, 3];
        let content = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(shape.clone(), content.clone()).unwrap();

        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.dimension, 2);
        assert_eq!(tensor.data.as_slice().unwrap(), content.as_slice());
    }

    #[test]
    fn new_fails_on_shape_content_mismatch() {
        let shape = vec![2, 3];
        let content = vec![1.0, 2.0, 3.0]; // Only 3 elements for 2x3 shape
        let result = Tensor::new(shape, content);

        assert!(result.is_err());
    }

    #[test]
    fn new_handles_empty_tensor() {
        let shape = vec![0];
        let content = vec![];
        let tensor = Tensor::new(shape.clone(), content).unwrap();

        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.dimension, 1);
        assert_eq!(tensor.data.len(), 0);
    }

    #[test]
    fn new_handles_scalar_tensor() {
        let shape = vec![1];
        let content = vec![42.0];
        let tensor = Tensor::new(shape, content).unwrap();

        assert_eq!(tensor.dimension, 1);
        assert_eq!(tensor.data[[0]], 42.0);
    }

    #[test]
    fn new_handles_3d_tensor() {
        let shape = vec![2, 2, 3];
        let content = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let tensor = Tensor::new(shape.clone(), content).unwrap();

        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.dimension, 3);
        assert_eq!(tensor.data.ndim(), 3);
    }

    #[test]
    fn zeros_creates_all_zeros() {
        let shape = vec![3, 4];
        let tensor = Tensor::zeros(shape.clone());

        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.dimension, 2);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn zeros_handles_single_dimension() {
        let shape = vec![5];
        let tensor = Tensor::zeros(shape.clone());

        assert_eq!(tensor.dimension, 1);
        assert_eq!(tensor.data.len(), 5);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn zeros_handles_large_tensor() {
        let shape = vec![100, 100];
        let tensor = Tensor::zeros(shape.clone());

        assert_eq!(tensor.data.len(), 10000);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn random_creates_values_in_correct_range() {
        let shape = vec![100, 100];
        let tensor = Tensor::random(shape);

        // random_range(0.0..100.0) generates values in [0.0, 100.0)
        assert!(tensor.data.iter().all(|&x| x >= 0.0 && x < 100.0));
        // Verify not all values are the same
        let first = tensor.data.iter().next().unwrap();
        assert!(tensor.data.iter().any(|&x| (x - first).abs() > 0.1));
    }

    #[test]
    fn random_creates_different_values() {
        let shape = vec![10, 10];
        let t1 = Tensor::random(shape.clone());
        let t2 = Tensor::random(shape);

        // Statistical test: at least 90% of values should differ
        let different_count = t1
            .data
            .iter()
            .zip(t2.data.iter())
            .filter(|(&a, &b)| (a - b).abs() > 0.001)
            .count();
        assert!(different_count > 90);
    }

    #[test]
    fn from_numpy_creates_tensor_from_array() {
        Python::attach(|py| {
            let arr = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0].into_pyarray(py);
            let arr_reshaped = arr.reshape([2, 3]).expect("reshape failed");
            let arr_dyn = arr_reshaped
                .as_any()
                .downcast::<numpy::PyArrayDyn<f32>>()
                .unwrap();
            let arr_readonly = arr_dyn.readonly();

            let tensor = Tensor::from_numpy(arr_readonly).unwrap();

            assert_eq!(tensor.shape, vec![2, 3]);
            assert_eq!(tensor.dimension, 2);
            assert_eq!(
                tensor.data.as_slice().unwrap(),
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            );
        });
    }

    #[test]
    fn from_numpy_preserves_shape() {
        Python::attach(|py| {
            let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
            let arr = data.into_pyarray(py);
            let arr_reshaped = arr.reshape([2, 3, 4]).expect("reshape failed");
            let arr_dyn = arr_reshaped
                .as_any()
                .downcast::<numpy::PyArrayDyn<f32>>()
                .unwrap();
            let arr_readonly = arr_dyn.readonly();

            let tensor = Tensor::from_numpy(arr_readonly).unwrap();

            assert_eq!(tensor.shape, vec![2, 3, 4]);
            assert_eq!(tensor.dimension, 3);
        });
    }
}

// =============================================================================
// TENSOR SHAPE OPERATIONS
// =============================================================================

#[cfg(test)]
mod tensor_shape_operations {
    use super::*;

    #[test]
    fn reshape_inplace_changes_shape_preserves_data() {
        let content = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut tensor = Tensor::new(vec![2, 6], content.clone()).unwrap();

        tensor.reshape_inplace(vec![3, 4]).unwrap();

        assert_eq!(tensor.shape, vec![3, 4]);
        assert_eq!(tensor.dimension, 2);
        assert_eq!(tensor.data.as_slice().unwrap(), content.as_slice());
    }

    #[test]
    fn reshape_inplace_fails_on_incompatible_size() {
        let mut tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = tensor.reshape_inplace(vec![2, 2]); // 6 elements can't fit 2x2

        assert!(result.is_err());
    }

    #[test]
    fn reshape_inplace_to_1d() {
        let mut tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        tensor.reshape_inplace(vec![6]).unwrap();

        assert_eq!(tensor.shape, vec![6]);
        assert_eq!(tensor.dimension, 1);
    }

    #[test]
    fn reshape_inplace_from_1d_to_multidimensional() {
        let mut tensor = Tensor::new(vec![24], (0..24).map(|x| x as f32).collect()).unwrap();
        tensor.reshape_inplace(vec![2, 3, 4]).unwrap();

        assert_eq!(tensor.shape, vec![2, 3, 4]);
        assert_eq!(tensor.dimension, 3);
        assert_eq!(tensor.data.len(), 24);
    }

    #[test]
    fn flatten_inplace_converts_to_1d() {
        let mut tensor = Tensor::new(vec![2, 3, 2], vec![1.0; 12]).unwrap();
        tensor.flatten_inplace();

        assert_eq!(tensor.shape, vec![12]);
        assert_eq!(tensor.dimension, 1);
        assert_eq!(tensor.data.ndim(), 1);
    }

    #[test]
    fn flatten_inplace_preserves_order() {
        let content = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut tensor = Tensor::new(vec![2, 3], content.clone()).unwrap();
        tensor.flatten_inplace();

        assert_eq!(tensor.data.as_slice().unwrap(), content.as_slice());
    }

    #[test]
    fn flatten_inplace_already_1d_is_noop() {
        let content = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor = Tensor::new(vec![4], content.clone()).unwrap();
        tensor.flatten_inplace();

        assert_eq!(tensor.shape, vec![4]);
        assert_eq!(tensor.data.as_slice().unwrap(), content.as_slice());
    }

    #[test]
    fn reshape_returns_new_tensor() {
        let content = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let tensor = Tensor::new(vec![2, 6], content.clone()).unwrap();

        let reshaped = tensor.reshape(vec![3, 4]).unwrap();

        assert_eq!(reshaped.shape, vec![3, 4]);
        assert_eq!(reshaped.dimension, 2);
        assert_eq!(reshaped.data.as_slice().unwrap(), content.as_slice());
        
        // Verifica che l'originale non sia stato modificato
        assert_eq!(tensor.shape, vec![2, 6]);
    }

    #[test]
    fn reshape_fails_on_incompatible_size() {
        let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = tensor.reshape(vec![2, 2]); // 6 elements can't fit 2x2

        assert!(result.is_err());
    }

    #[test]
    fn flatten_returns_new_1d_tensor() {
        let tensor = Tensor::new(vec![2, 3, 2], vec![1.0; 12]).unwrap();
        let flattened = tensor.flatten();

        assert_eq!(flattened.shape, vec![12]);
        assert_eq!(flattened.dimension, 1);
        assert_eq!(flattened.data.ndim(), 1);
        
        // Verifica che l'originale non sia stato modificato
        assert_eq!(tensor.shape, vec![2, 3, 2]);
    }

    #[test]
    fn flatten_preserves_order() {
        let content = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(vec![2, 3], content.clone()).unwrap();
        let flattened = tensor.flatten();

        assert_eq!(flattened.data.as_slice().unwrap(), content.as_slice());
    }


    #[test]
    fn transpose_2d_swaps_dimensions() {
        let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let transposed = tensor.transpose().unwrap();

        assert_eq!(transposed.shape, vec![3, 2]);
        assert_eq!(transposed.dimension, 2);
        // Original: [[1,2,3], [4,5,6]]
        // Transposed: [[1,4], [2,5], [3,6]]
        assert_eq!(transposed.data[[0, 0]], 1.0);
        assert_eq!(transposed.data[[0, 1]], 4.0);
        assert_eq!(transposed.data[[1, 0]], 2.0);
        assert_eq!(transposed.data[[1, 1]], 5.0);
        assert_eq!(transposed.data[[2, 0]], 3.0);
        assert_eq!(transposed.data[[2, 1]], 6.0);
    }

    #[test]
    fn transpose_square_matrix() {
        let tensor = Tensor::new(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let transposed = tensor.transpose().unwrap();

        assert_eq!(transposed.shape, vec![3, 3]);
        assert_eq!(transposed.data[[0, 0]], 1.0);
        assert_eq!(transposed.data[[0, 1]], 4.0);
        assert_eq!(transposed.data[[0, 2]], 7.0);
    }

    #[test]
    fn transpose_fails_on_non_2d() {
        let tensor = Tensor::new(vec![2, 3, 4], vec![1.0; 24]).unwrap();
        let result = tensor.transpose();

        assert!(result.is_err());
    }

    #[test]
    fn transpose_fails_on_1d() {
        let tensor = Tensor::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let result = tensor.transpose();

        assert!(result.is_err());
    }
}

// =============================================================================
// TENSOR ARITHMETIC
// =============================================================================

#[cfg(test)]
mod tensor_arithmetic {
    use super::*;

    #[test]
    fn add_tensors_same_shape() {
        let t1 = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let t2 = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = (&t1 + &t2).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn add_tensors_incompatible_shape_fails() {
        let t1 = Tensor::new(vec![2, 2], vec![1.0; 4]).unwrap();
        let t2 = Tensor::new(vec![3, 3], vec![1.0; 9]).unwrap();
        let result = &t1 + &t2;

        assert!(result.is_err());
    }

    #[test]
    fn add_scalar() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = &t + 10.0;

        assert_eq!(result.data.as_slice().unwrap(), &[11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn add_negative_scalar() {
        let t = Tensor::new(vec![2], vec![5.0, 10.0]).unwrap();
        let result = &t + (-3.0);

        assert_eq!(result.data.as_slice().unwrap(), &[2.0, 7.0]);
    }

    #[test]
    fn sub_tensors() {
        let t1 = Tensor::new(vec![2], vec![10.0, 20.0]).unwrap();
        let t2 = Tensor::new(vec![2], vec![3.0, 7.0]).unwrap();
        let result = (&t1 - &t2).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[7.0, 13.0]);
    }

    #[test]
    fn sub_scalar() {
        let t = Tensor::new(vec![3], vec![5.0, 10.0, 15.0]).unwrap();
        let result = &t - 5.0;

        assert_eq!(result.data.as_slice().unwrap(), &[0.0, 5.0, 10.0]);
    }

    #[test]
    fn sub_results_in_negative() {
        let t1 = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let t2 = Tensor::new(vec![2], vec![3.0, 4.0]).unwrap();
        let result = (&t1 - &t2).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[-2.0, -2.0]);
    }

    #[test]
    fn mul_tensors_elementwise() {
        let t1 = Tensor::new(vec![2, 2], vec![2.0, 3.0, 4.0, 5.0]).unwrap();
        let t2 = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = (&t1 * &t2).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn mul_scalar() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let result = &t * 2.0;

        assert_eq!(result.data.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn mul_by_zero() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let result = &t * 0.0;

        assert_eq!(result.data.as_slice().unwrap(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn div_tensors() {
        let t1 = Tensor::new(vec![2], vec![10.0, 20.0]).unwrap();
        let t2 = Tensor::new(vec![2], vec![2.0, 4.0]).unwrap();
        let result = (&t1 / &t2).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[5.0, 5.0]);
    }

    #[test]
    fn div_scalar() {
        let t = Tensor::new(vec![2], vec![10.0, 20.0]).unwrap();
        let result = (&t / 2.0).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[5.0, 10.0]);
    }

    #[test]
    fn div_by_zero_scalar_fails() {
        let t = Tensor::new(vec![2], vec![10.0, 20.0]).unwrap();
        let result = &t / 0.0;

        assert!(result.is_err());
    }

    #[test]
    fn div_tensor_with_zero_element_fails() {
        let t1 = Tensor::new(vec![2], vec![10.0, 20.0]).unwrap();
        let t2 = Tensor::new(vec![2], vec![2.0, 0.0]).unwrap();
        let result = &t1 / &t2;

        assert!(result.is_err());
    }

    #[test]
    fn chained_operations() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let result = ((&t + 1.0) * 2.0) - 1.0;

        // (t + 1) * 2 - 1 = ([2,3,4] * 2) - 1 = [4,6,8] - 1 = [3,5,7]
        assert_eq!(result.data.as_slice().unwrap(), &[3.0, 5.0, 7.0]);
    }
}

// =============================================================================
// TENSOR DOT PRODUCT
// =============================================================================

#[cfg(test)]
mod tensor_dot_product {
    use super::*;

    #[test]
    fn dot_1d_vectors_returns_scalar() {
        let t1 = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let t2 = Tensor::new(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
        let result = t1.dot(&t2).unwrap();

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result.dimension, 0);
        assert_eq!(result.shape, Vec::<usize>::new());
        // Access 0D scalar array directly
        assert_eq!(result.data[[]], 32.0);
    }

    #[test]
    fn dot_1d_incompatible_size_fails() {
        let t1 = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let t2 = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = t1.dot(&t2);

        assert!(result.is_err());
    }

    #[test]
    fn dot_1d_with_2d_matrix_vector() {
        let vec = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let mat = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = vec.dot(&mat).unwrap();

        // [1,2,3] × [[1,2], [3,4], [5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        assert_eq!(result.shape, vec![2]);
        assert_eq!(result.dimension, 1);
        assert_eq!(result.data.as_slice().unwrap(), &[22.0, 28.0]);
    }

    #[test]
    fn dot_2d_with_1d_matrix_vector() {
        let mat = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vec = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let result = mat.dot(&vec).unwrap();

        // [[1,2,3], [4,5,6]] × [1,2,3] = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
        assert_eq!(result.shape, vec![2]);
        assert_eq!(result.dimension, 1);
        assert_eq!(result.data.as_slice().unwrap(), &[14.0, 32.0]);
    }

    #[test]
    fn dot_2d_matrix_multiplication() {
        let m1 = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let m2 = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let result = m1.dot(&m2).unwrap();

        // [[1,2,3], [4,5,6]] × [[7,8], [9,10], [11,12]]
        // = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[58, 64], [139, 154]]
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.dimension, 2);
        assert_eq!(result.data.as_slice().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn dot_2d_incompatible_dimensions_fails() {
        let m1 = Tensor::new(vec![2, 3], vec![1.0; 6]).unwrap();
        let m2 = Tensor::new(vec![2, 2], vec![1.0; 4]).unwrap();
        let result = m1.dot(&m2);

        // 2×3 can't multiply with 2×2 (inner dimensions don't match)
        assert!(result.is_err());
    }

    #[test]
    fn dot_identity_matrix_preserves_input() {
        let m = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let identity = Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let result = m.dot(&identity).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), m.data.as_slice().unwrap());
    }

    #[test]
    fn dot_large_matrices_uses_parallel() {
        // This test verifies parallel matmul works for large matrices
        let m1 = Tensor::random(vec![100, 100]);
        let m2 = Tensor::random(vec![100, 100]);
        let result = m1.dot(&m2);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.shape, vec![100, 100]);
        assert_eq!(result.dimension, 2);
    }

    #[test]
    fn dot_square_matrices_various_sizes() {
        for size in [2, 5, 10, 20] {
            let m1 = Tensor::zeros(vec![size, size]);
            let m2 = Tensor::zeros(vec![size, size]);
            let result = m1.dot(&m2).unwrap();

            assert_eq!(result.shape, vec![size, size]);
            assert!(result.data.iter().all(|&x| x == 0.0));
        }
    }
}

// =============================================================================
// TENSOR CONCATENATION
// =============================================================================

#[cfg(test)]
mod tensor_concatenation {
    use super::*;

    #[test]
    fn push_appends_along_axis() {
        let mut t1 = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let t2 = Tensor::new(vec![2], vec![3.0, 4.0]).unwrap();
        t1.push(&t2, 0).unwrap();

        assert_eq!(t1.shape, vec![4]);
        assert_eq!(t1.data.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn push_2d_tensors_along_rows() {
        let mut t1 = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let t2 = Tensor::new(vec![1, 2], vec![5.0, 6.0]).unwrap();
        t1.push(&t2, 0).unwrap();

        assert_eq!(t1.shape, vec![3, 2]);
    }

    #[test]
    fn push_row_extends_2d_matrix() {
        let mut mat = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let row = Tensor::new(vec![3], vec![7.0, 8.0, 9.0]).unwrap();
        mat.push_row(&row).unwrap();

        assert_eq!(mat.shape, vec![3, 3]);
        assert_eq!(mat.dimension, 2);
        assert_eq!(mat.data[[2, 0]], 7.0);
        assert_eq!(mat.data[[2, 1]], 8.0);
        assert_eq!(mat.data[[2, 2]], 9.0);
    }

    #[test]
    fn push_row_converts_1d_to_2d() {
        let mut vec = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let row = Tensor::new(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
        vec.push_row(&row).unwrap();

        assert_eq!(vec.shape, vec![2, 3]);
        assert_eq!(vec.dimension, 2);
    }

    #[test]
    fn push_row_fails_on_width_mismatch() {
        let mut mat = Tensor::new(vec![2, 3], vec![1.0; 6]).unwrap();
        let row = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let result = mat.push_row(&row);

        assert!(result.is_err());
    }

    #[test]
    fn push_row_fails_on_non_1d_row() {
        let mut mat = Tensor::new(vec![2, 2], vec![1.0; 4]).unwrap();
        let non_1d = Tensor::new(vec![1, 2], vec![1.0, 2.0]).unwrap();
        let result = mat.push_row(&non_1d);

        assert!(result.is_err());
    }

    #[test]
    fn push_column_extends_2d_matrix() {
        let mut mat = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let col = Tensor::new(vec![2], vec![5.0, 6.0]).unwrap();
        mat.push_column(&col).unwrap();

        assert_eq!(mat.shape, vec![2, 3]);
        assert_eq!(mat.dimension, 2);
        assert_eq!(mat.data[[0, 2]], 5.0);
        assert_eq!(mat.data[[1, 2]], 6.0);
    }

    #[test]
    fn push_column_converts_1d_to_2d() {
        let mut vec = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let col = Tensor::new(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
        vec.push_column(&col).unwrap();

        assert_eq!(vec.shape, vec![3, 2]);
        assert_eq!(vec.dimension, 2);
    }

    #[test]
    fn push_column_fails_on_height_mismatch() {
        let mut mat = Tensor::new(vec![2, 2], vec![1.0; 4]).unwrap();
        let col = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let result = mat.push_column(&col);

        assert!(result.is_err());
    }

    #[test]
    fn cat_concatenates_multiple_tensors() {
        let t1 = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let t2 = Tensor::new(vec![2], vec![3.0, 4.0]).unwrap();
        let t3 = Tensor::new(vec![2], vec![5.0, 6.0]).unwrap();
        let result = Tensor::cat(vec![t1, t2, t3], 0).unwrap();

        assert_eq!(result.shape, vec![6]);
        assert_eq!(
            result.data.as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn cat_single_tensor_returns_copy() {
        let t = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let result = Tensor::cat(vec![t.clone()], 0).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), t.data.as_slice().unwrap());
    }

    #[test]
    fn cat_inplace_concatenates() {
        let t1 = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let t2 = Tensor::new(vec![2], vec![3.0, 4.0]).unwrap();
        let t3 = Tensor::new(vec![2], vec![5.0, 6.0]).unwrap();
        let result = t1.cat_inplace(vec![t2, t3], 0).unwrap();

        assert_eq!(result.shape, vec![6]);
        assert_eq!(
            result.data.as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }
}

// =============================================================================
// TENSOR SERIALIZATION
// =============================================================================

#[cfg(test)]
mod tensor_serialization {
    use super::*;

    #[test]
    fn to_dict_from_dict_roundtrip() {
        Python::attach(|py| {
            let original = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
            let dict = original.to_dict().unwrap();
            let dict_bound = dict.bind(py);
            let restored = Tensor::from_dict(dict_bound.clone()).unwrap();

            assert_eq!(restored.shape, original.shape);
            assert_eq!(restored.dimension, original.dimension);
            assert_eq!(
                restored.data.as_slice().unwrap(),
                original.data.as_slice().unwrap()
            );
        });
    }

    #[test]
    fn to_dict_preserves_all_fields() {
        Python::attach(|py| {
            let tensor = Tensor::new(vec![2, 3, 4], vec![1.0; 24]).unwrap();
            let dict = tensor.to_dict().unwrap();
            let dict_bound = dict.bind(py);

            let dimension: usize = dict_bound
                .get_item("dimension")
                .expect("dimension missing")
                .extract()
                .unwrap();
            assert_eq!(dimension, 3);

            let shape: Vec<usize> = dict_bound
                .get_item("shape")
                .expect("shape missing")
                .extract()
                .unwrap();
            assert_eq!(shape, vec![2, 3, 4]);

            let data: Vec<f32> = dict_bound
                .get_item("data")
                .expect("data missing")
                .extract()
                .unwrap();
            assert_eq!(data.len(), 24);
        });
    }

    #[test]
    fn from_dict_validates_shape_data_consistency() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("dimension", 2).unwrap();
            dict.set_item("shape", vec![2, 2]).unwrap();
            dict.set_item("data", vec![1.0_f32, 2.0, 3.0]).unwrap(); // Only 3 elements for 2x2

            let result = Tensor::from_dict(dict.into_any());

            assert!(result.is_err());
        });
    }

    #[test]
    fn from_dict_rejects_missing_fields() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("dimension", 2).unwrap();
            // Missing "shape" and "data"

            let result = Tensor::from_dict(dict.into_any());

            assert!(result.is_err());
        });
    }

    #[test]
    fn serialization_preserves_f32_precision() {
        Python::attach(|py| {
            let original =
                Tensor::new(vec![3], vec![1.234567_f32, 9.876543_f32, 0.000001_f32]).unwrap();
            let dict = original.to_dict().unwrap();
            let restored = Tensor::from_dict(dict.bind(py).clone()).unwrap();

            for (a, b) in original.data.iter().zip(restored.data.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        });
    }
}

// =============================================================================
// TENSOR ITERATION
// =============================================================================

#[cfg(test)]
mod tensor_iteration {
    use super::*;

    #[test]
    fn iter_yields_all_elements_in_order() {
        Python::attach(|py| {
            let content = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let tensor = Tensor::new(vec![2, 3], content.clone()).unwrap();
            let tensor_py = Py::new(py, tensor).unwrap();

            // Get iterator via __iter__
            let iter_obj = tensor_py.bind(py).call_method0("__iter__").unwrap();

            // Collect values via __next__
            let mut collected = vec![];
            loop {
                match iter_obj.call_method0("__next__") {
                    Ok(val) => collected.push(val.extract::<f32>().unwrap()),
                    Err(_) => break, // StopIteration
                }
            }

            assert_eq!(collected, content);
        });
    }

    #[test]
    fn iter_empty_tensor_stops_immediately() {
        Python::attach(|py| {
            let tensor = Tensor::new(vec![0], vec![]).unwrap();
            let tensor_py = Py::new(py, tensor).unwrap();

            let iter_obj = tensor_py.bind(py).call_method0("__iter__").unwrap();

            // First __next__ should raise StopIteration
            let result = iter_obj.call_method0("__next__");
            assert!(result.is_err());
        });
    }

    #[test]
    fn iter_single_element() {
        Python::attach(|py| {
            let tensor = Tensor::new(vec![1], vec![42.0]).unwrap();
            let tensor_py = Py::new(py, tensor).unwrap();

            let iter_obj = tensor_py.bind(py).call_method0("__iter__").unwrap();

            let first = iter_obj
                .call_method0("__next__")
                .unwrap()
                .extract::<f32>()
                .unwrap();
            assert_eq!(first, 42.0);

            // Second call should raise StopIteration
            let result = iter_obj.call_method0("__next__");
            assert!(result.is_err());
        });
    }

    #[test]
    fn iter_multidimensional_flattens() {
        Python::attach(|py| {
            let tensor = Tensor::new(vec![2, 2, 3], vec![1.0; 12]).unwrap();
            let tensor_py = Py::new(py, tensor).unwrap();

            let iter_obj = tensor_py.bind(py).call_method0("__iter__").unwrap();

            let mut count = 0;
            loop {
                match iter_obj.call_method0("__next__") {
                    Ok(val) => {
                        assert_eq!(val.extract::<f32>().unwrap(), 1.0);
                        count += 1;
                    }
                    Err(_) => break,
                }
            }

            assert_eq!(count, 12);
        });
    }

    #[test]
    fn iter_via_python_for_loop() {
        Python::attach(|py| {
            let tensor = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
            let tensor_py = Py::new(py, tensor).unwrap();

            py_run!(
                py,
                tensor_py,
                r#"
result = []
for x in tensor_py:
    result.append(x)
assert result == [1.0, 2.0, 3.0], f"Expected [1.0, 2.0, 3.0], got {result}"
            "#
            );
        });
    }
}

// =============================================================================
// TENSOR EDGE CASES
// =============================================================================

#[cfg(test)]
mod tensor_edge_cases {
    use super::*;

    #[test]
    fn negative_values_preserved() {
        let t = Tensor::new(vec![2], vec![-1.0, -2.0]).unwrap();
        assert_eq!(t.data.as_slice().unwrap(), &[-1.0, -2.0]);
    }

    #[test]
    fn very_small_values() {
        let t = Tensor::new(vec![2], vec![1e-10, 1e-20]).unwrap();
        assert_eq!(t.data[[0]], 1e-10);
        assert_eq!(t.data[[1]], 1e-20);
    }

    #[test]
    fn very_large_values() {
        let t = Tensor::new(vec![2], vec![1e10, 1e20]).unwrap();
        assert_eq!(t.data[[0]], 1e10);
        assert_eq!(t.data[[1]], 1e20);
    }

    #[test]
    fn mixed_positive_negative_operations() {
        let t1 = Tensor::new(vec![2], vec![1.0, -1.0]).unwrap();
        let t2 = Tensor::new(vec![2], vec![-2.0, 2.0]).unwrap();
        let result = (&t1 + &t2).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[-1.0, 1.0]);
    }

    #[test]
    fn large_tensor_operations_complete() {
        let shape = vec![1000];
        let t1 = Tensor::zeros(shape.clone());
        let t2 = Tensor::random(shape);
        let result = (&t1 + &t2).unwrap();

        assert_eq!(result.data.len(), 1000);
        assert_eq!(result.shape, vec![1000]);
    }

    #[test]
    fn f32_precision_limits_addition() {
        let t = Tensor::new(vec![1], vec![1.0]).unwrap();
        let result = &t + 1e-6;

        // f32 has ~7 significant digits, 1 + 1e-6 should be distinguishable
        assert!(result.data[[0]] > 1.0);
    }

    #[test]
    fn f32_precision_limits_large_numbers() {
        let t = Tensor::new(vec![1], vec![1e20]).unwrap();
        let result = &t + 1.0;

        // Adding 1.0 to 1e20 is below f32 precision
        assert_eq!(result.data[[0]], t.data[[0]]);
    }

    #[test]
    fn length_method_returns_total_elements() {
        let t = Tensor::new(vec![2, 3, 4], vec![1.0; 24]).unwrap();
        assert_eq!(t.length(), 24);
    }

    #[test]
    fn repr_includes_shape_and_dimension() {
        Python::attach(|py| {
            let tensor = Tensor::new(vec![2, 3], vec![1.0; 6]).unwrap();
            let tensor_py = Py::new(py, tensor).unwrap();

            py_run!(
                py,
                tensor_py,
                r#"
repr_str = repr(tensor_py)
assert "dimension=2" in repr_str, f"Missing dimension in {repr_str}"
assert "shape=[2, 3]" in repr_str, f"Missing shape in {repr_str}"
            "#
            );
        });
    }

    #[test]
    fn get_data_returns_numpy_array() {
        Python::attach(|py| {
            let tensor = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
            let tensor_py = Py::new(py, tensor).unwrap();

            py_run!(
                py,
                tensor_py,
                r#"
import numpy as np
arr = tensor_py.data
assert isinstance(arr, np.ndarray), f"Expected ndarray, got {type(arr)}"
assert arr.shape == (2, 2), f"Expected (2, 2), got {arr.shape}"
assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
            "#
            );
        });
    }

    #[test]
    fn set_data_updates_tensor() {
        Python::attach(|py| {
            let tensor = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
            let tensor_py = Py::new(py, tensor).unwrap();

            py_run!(
                py,
                tensor_py,
                r#"
import numpy as np
new_data = np.array([10.0, 20.0], dtype=np.float32)
tensor_py.data = new_data
assert tensor_py.data[0] == 10.0, f"Expected 10.0, got {tensor_py.data[0]}"
assert tensor_py.data[1] == 20.0, f"Expected 20.0, got {tensor_py.data[1]}"
            "#
            );
        });
    }
}
