#[cfg(test)]
mod tensor_construction {
    use crate::structures::tensor::Tensor;
    use ndarray::ArrayD;

    #[test]
    fn new_creates_tensor_with_correct_shape_and_content() {
        let shape = vec![2, 3];
        let content = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(shape.clone(), content.clone()).unwrap();

        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.data.as_slice().unwrap(), content.as_slice());
    }

    #[test]
    fn new_fails_on_shape_content_mismatch() {
        let shape = vec![2, 3];
        let content = vec![1.0, 2.0, 3.0];
        let result = Tensor::new(shape, content);

        assert!(result.is_err());
    }

    #[test]
    fn new_handles_empty_tensor() {
        let shape = vec![0];
        let content = vec![];
        let tensor = Tensor::new(shape.clone(), content).unwrap();

        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.data.len(), 0);
    }

    #[test]
    fn new_handles_scalar_tensor() {
        let shape = vec![1];
        let content = vec![42.0];
        let tensor = Tensor::new(shape, content).unwrap();

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
        assert_eq!(tensor.data.ndim(), 3);
    }

    #[test]
    fn zeros_creates_all_zeros() {
        let shape = vec![3, 4];
        let tensor = Tensor::zeros(shape.clone()).unwrap();

        assert_eq!(tensor.shape, shape);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn zeros_handles_single_dimension() {
        let shape = vec![5];
        let tensor = Tensor::zeros(shape).unwrap();

        assert_eq!(tensor.data.len(), 5);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn random_creates_values_in_range() {
        let shape = vec![100, 100];
        let tensor = Tensor::random(shape).unwrap();

        assert!(tensor.data.iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn random_creates_different_values() {
        let shape = vec![10, 10];
        let t1 = Tensor::random(shape.clone()).unwrap();
        let t2 = Tensor::random(shape).unwrap();

        assert_ne!(t1.data.as_slice().unwrap(), t2.data.as_slice().unwrap());
    }
}

#[cfg(test)]
mod tensor_shape_operations {
    use crate::structures::tensor::Tensor;

    #[test]
    fn reshape_changes_shape_preserves_data() {
        let shape = vec![2, 6];
        let content = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut tensor = Tensor::new(shape, content.clone()).unwrap();

        tensor.reshape(vec![3, 4]).unwrap();

        assert_eq!(tensor.shape, vec![3, 4]);
        assert_eq!(tensor.data.as_slice().unwrap(), content.as_slice());
    }

    #[test]
    fn reshape_fails_on_incompatible_size() {
        let mut tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = tensor.reshape(vec![2, 2]);

        assert!(result.is_err());
    }

    #[test]
    fn reshape_to_1d() {
        let mut tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        tensor.reshape(vec![6]).unwrap();

        assert_eq!(tensor.shape, vec![6]);
    }

    #[test]
    fn flatten_converts_to_1d() {
        let tensor = Tensor::new(vec![2, 3, 2], vec![1.0; 12]).unwrap();
        let flattened = tensor.flatten().unwrap();

        assert_eq!(flattened.shape, vec![12]);
        assert_eq!(flattened.data.ndim(), 1);
    }

    #[test]
    fn flatten_preserves_order() {
        let content = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(vec![2, 3], content.clone()).unwrap();
        let flattened = tensor.flatten().unwrap();

        assert_eq!(flattened.data.as_slice().unwrap(), content.as_slice());
    }

    #[test]
    fn transpose_2d_swaps_dimensions() {
        let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let transposed = tensor.transpose().unwrap();

        assert_eq!(transposed.shape, vec![3, 2]);
        assert_eq!(transposed.data[[0, 0]], 1.0);
        assert_eq!(transposed.data[[0, 1]], 4.0);
        assert_eq!(transposed.data[[1, 0]], 2.0);
    }

    #[test]
    fn transpose_fails_on_non_2d() {
        let tensor = Tensor::new(vec![2, 3, 4], vec![1.0; 24]).unwrap();
        let result = tensor.transpose();

        assert!(result.is_err());
    }

    #[test]
    fn transpose_1d_fails() {
        let tensor = Tensor::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let result = tensor.transpose();

        assert!(result.is_err());
    }
}

#[cfg(test)]
mod tensor_arithmetic {
    use crate::structures::tensor::Tensor;

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
    fn div_by_zero_fails() {
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
}

#[cfg(test)]
mod tensor_dot_product {
    use crate::structures::tensor::Tensor;

    #[test]
    fn dot_1d_vectors() {
        let t1 = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let t2 = Tensor::new(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
        let result = t1.dot(&t2).unwrap();

        assert_eq!(result.shape, vec![1]);
        assert_eq!(result.data[[0]], 32.0);
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

        assert_eq!(result.shape, vec![2]);
        assert_eq!(result.data.as_slice().unwrap(), &[22.0, 28.0]);
    }

    #[test]
    fn dot_2d_with_1d_matrix_vector() {
        let mat = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vec = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let result = mat.dot(&vec).unwrap();

        assert_eq!(result.shape, vec![2]);
        assert_eq!(result.data.as_slice().unwrap(), &[14.0, 32.0]);
    }

    #[test]
    fn dot_2d_matrix_multiplication() {
        let m1 = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let m2 = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let result = m1.dot(&m2).unwrap();

        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data.as_slice().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn dot_2d_incompatible_dimensions_fails() {
        let m1 = Tensor::new(vec![2, 3], vec![1.0; 6]).unwrap();
        let m2 = Tensor::new(vec![2, 2], vec![1.0; 4]).unwrap();
        let result = m1.dot(&m2);

        assert!(result.is_err());
    }

    #[test]
    fn dot_identity_matrix() {
        let m = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let identity = Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let result = m.dot(&identity).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), m.data.as_slice().unwrap());
    }

    #[test]
    fn dot_large_matrices_performance() {
        let m1 = Tensor::random(vec![100, 100]).unwrap();
        let m2 = Tensor::random(vec![100, 100]).unwrap();
        let result = m1.dot(&m2);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().shape, vec![100, 100]);
    }
}

#[cfg(test)]
mod tensor_concatenation {
    use crate::structures::tensor::Tensor;

    #[test]
    fn push_appends_1d_tensor() {
        let mut t1 = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let t2 = Tensor::new(vec![2], vec![3.0, 4.0]).unwrap();
        t1.push(&t2).unwrap();

        assert_eq!(t1.shape, vec![2, 2]);
        assert_eq!(t1.data.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn push_fails_on_shape_mismatch() {
        let mut t1 = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let t2 = Tensor::new(vec![3], vec![3.0, 4.0, 5.0]).unwrap();
        let result = t1.push(&t2);

        assert!(result.is_err());
    }

    #[test]
    fn push_row_extends_2d() {
        let mut mat = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let row = Tensor::new(vec![3], vec![7.0, 8.0, 9.0]).unwrap();
        mat.push_row(&row).unwrap();

        assert_eq!(mat.shape, vec![3, 3]);
        assert_eq!(mat.data[[2, 0]], 7.0);
        assert_eq!(mat.data[[2, 2]], 9.0);
    }

    #[test]
    fn push_row_fails_on_width_mismatch() {
        let mut mat = Tensor::new(vec![2, 3], vec![1.0; 6]).unwrap();
        let row = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let result = mat.push_row(&row);

        assert!(result.is_err());
    }

    #[test]
    fn push_column_extends_2d() {
        let mut mat = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let col = Tensor::new(vec![2], vec![5.0, 6.0]).unwrap();
        mat.push_column(&col).unwrap();

        assert_eq!(mat.shape, vec![2, 3]);
        assert_eq!(mat.data[[0, 2]], 5.0);
        assert_eq!(mat.data[[1, 2]], 6.0);
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
        let result = Tensor::cat(vec![&t1, &t2, &t3]).unwrap();

        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(
            result.data.as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn cat_fails_on_empty_list() {
        let result = Tensor::cat(vec![]);

        assert!(result.is_err());
    }

    #[test]
    fn cat_single_tensor_returns_copy() {
        let t = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let result = Tensor::cat(vec![&t]).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), t.data.as_slice().unwrap());
    }

    #[test]
    fn cat_fails_on_shape_mismatch() {
        let t1 = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let t2 = Tensor::new(vec![3], vec![3.0, 4.0, 5.0]).unwrap();
        let result = Tensor::cat(vec![&t1, &t2]);

        assert!(result.is_err());
    }
}

#[cfg(test)]
mod tensor_serialization {
    use crate::structures::tensor::Tensor;
    use pyo3::Python;

    #[test]
    fn to_dict_from_dict_roundtrip() {
        Python::with_gil(|py| {
            let original = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
            let dict = original.to_dict(py).unwrap();
            let restored = Tensor::from_dict(dict).unwrap();

            assert_eq!(restored.shape, original.shape);
            assert_eq!(
                restored.data.as_slice().unwrap(),
                original.data.as_slice().unwrap()
            );
        });
    }

    #[test]
    fn to_dict_preserves_shape() {
        Python::with_gil(|py| {
            let tensor = Tensor::new(vec![2, 3, 4], vec![1.0; 24]).unwrap();
            let dict = tensor.to_dict(py).unwrap();

            let shape = dict.get_item("shape").unwrap().unwrap();
            let shape_list: Vec<usize> = shape.extract().unwrap();
            assert_eq!(shape_list, vec![2, 3, 4]);
        });
    }

    #[test]
    fn from_dict_validates_shape() {
        Python::with_gil(|py| {
            let original = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
            let mut dict = original.to_dict(py).unwrap();

            dict.set_item("shape", vec![3]).unwrap();
            let result = Tensor::from_dict(dict);

            assert!(result.is_err());
        });
    }
}

#[cfg(test)]
mod tensor_iteration {
    use crate::structures::tensor::Tensor;
    use pyo3::Python;

    #[test]
    fn iter_yields_all_elements() {
        Python::with_gil(|py| {
            let content = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let tensor = Tensor::new(vec![2, 3], content.clone()).unwrap();
            let mut iter = tensor.__iter__();

            let mut collected = vec![];
            while let Some(val) = iter.__next__() {
                collected.push(val);
            }

            assert_eq!(collected, content);
        });
    }

    #[test]
    fn iter_empty_tensor() {
        Python::with_gil(|_py| {
            let tensor = Tensor::new(vec![0], vec![]).unwrap();
            let mut iter = tensor.__iter__();

            assert!(iter.__next__().is_none());
        });
    }

    #[test]
    fn iter_single_element() {
        Python::with_gil(|_py| {
            let tensor = Tensor::new(vec![1], vec![42.0]).unwrap();
            let mut iter = tensor.__iter__();

            assert_eq!(iter.__next__(), Some(42.0));
            assert!(iter.__next__().is_none());
        });
    }
}

#[cfg(test)]
mod tensor_edge_cases {
    use crate::structures::tensor::Tensor;

    #[test]
    fn negative_values() {
        let t = Tensor::new(vec![2], vec![-1.0, -2.0]).unwrap();
        assert_eq!(t.data.as_slice().unwrap(), &[-1.0, -2.0]);
    }

    #[test]
    fn very_small_values() {
        let t = Tensor::new(vec![2], vec![1e-10, 1e-20]).unwrap();
        assert_eq!(t.data[[0]], 1e-10);
    }

    #[test]
    fn very_large_values() {
        let t = Tensor::new(vec![2], vec![1e10, 1e20]).unwrap();
        assert_eq!(t.data[[0]], 1e10);
    }

    #[test]
    fn mixed_positive_negative() {
        let t1 = Tensor::new(vec![2], vec![1.0, -1.0]).unwrap();
        let t2 = Tensor::new(vec![2], vec![-2.0, 2.0]).unwrap();
        let result = (&t1 + &t2).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[-1.0, 1.0]);
    }

    #[test]
    fn large_tensor_operations() {
        let shape = vec![1000];
        let t1 = Tensor::zeros(shape.clone()).unwrap();
        let t2 = Tensor::random(shape).unwrap();
        let result = (&t1 + &t2).unwrap();

        assert_eq!(result.data.len(), 1000);
    }

    #[test]
    fn f32_precision_limits() {
        let t = Tensor::new(vec![1], vec![1.0]).unwrap();
        let result = &t + 1e-10;

        assert!(result.data[[0]] > 1.0);
    }
}
