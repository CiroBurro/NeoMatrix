//! Parallel matrix multiplication using Rayon.
//!
//! This module provides high-performance 2D matrix multiplication leveraging Rayon's
//! parallel iterators to distribute computation across available CPU cores.

use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView2, Axis};

/// Performs parallel matrix multiplication: `C = A × B`.
///
/// # Algorithm
///
/// Computes the matrix product `C[i,j] = Σ A[i,k] * B[k,j]` for all i, j.
/// Parallelizes across rows of the output matrix, with each thread computing
/// one or more complete rows independently.
///
/// # Parameters
///
/// - `t_1`: Left matrix A with shape `(m, n)` (as ArrayView2 for zero-copy)
/// - `t_2`: Right matrix B with shape `(n, p)` (as ArrayView2 for zero-copy)
///
/// # Returns
///
/// A new `Array2<f32>` with shape `(m, p)` containing the matrix product.
///
/// # Performance
///
/// - **Parallelization**: Row-level parallelism via Rayon's `par_iter()`
/// - **Cache efficiency**: Each thread computes full rows to maximize locality
/// - **Thread count**: Automatically determined by Rayon based on available cores
///
/// # Panics
///
/// Panics if the inner dimensions don't match (A.cols != B.rows). This is
/// enforced by ndarray's `dot()` method used internally.
///
/// # Example
///
/// ```ignore
/// use ndarray::Array2;
/// use neomatrix_core::math::matmul::par_dot;
///
/// let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let b = Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
/// let c = par_dot(a.view(), b.view()); // Shape: (2, 2)
/// ```
pub(crate) fn par_dot(t_1: ArrayView2<f32>, t_2: ArrayView2<f32>) -> Array2<f32> {
    let (m, _) = t_1.dim();
    let (_, p) = t_2.dim();

    // Initialize result matrix with zeros
    let mut a = Array2::zeros((m, p));

    // Parallel iteration over rows of the result matrix
    // Each thread computes: C[i, :] = A[i, :] × B
    a.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            // Compute each element in the row: C[i,j] = A[i,:] · B[:,j]
            for j in 0..p {
                row[j] = t_1.row(i).dot(&t_2.column(j));
            }
        });

    a
}
