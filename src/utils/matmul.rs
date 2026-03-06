//! Parallel matrix multiplication using Rayon for multi-threaded performance.
//!
//! Provides `par_dot` for 2D×2D matrix multiplication with row-level parallelization.
//! Uses zero-copy `ArrayView2` to avoid unnecessary data duplication.
//!
//! # Performance
//!
//! - **Parallelization**: Rayon threads over output rows
//! - **Algorithm**: Row(i) · Column(j) for each element
//! - **Memory**: Zero-copy via `ArrayView2` (no clone on input)
//! - **Dimensions**: Assumes compatible shapes (n×m) · (m×p) → (n×p)
//!
//! # Example
//!
//! ```rust
//! use ndarray::Array2;
//! use neomatrix::utils::matmul::par_dot;
//!
//! let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
//! let b = Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
//! let c = par_dot(a.view(), b.view());  // 2×2 result
//! ```

use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView2, Axis};

pub fn par_dot(t_1: ArrayView2<f32>, t_2: ArrayView2<f32>) -> Array2<f32> {
    let (m, _) = t_1.dim();
    let (_, p) = t_2.dim();

    let mut a = Array2::zeros((m, p));

    a.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..p {
                row[j] = t_1.row(i).dot(&t_2.column(j));
            }
        });

    a
}
