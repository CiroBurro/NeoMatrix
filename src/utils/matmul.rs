/// This module provides a function for executing matrix multiplication in parallel using Rayon crate
/// Necessary imports
use ndarray::Array2;
use ndarray::Axis;
use ndarray::parallel::prelude::*;

/// Function to execute the dot product of two matrices in parallel
/// ! Not a Python function !
/// 
/// # Arguments
/// * `t_1` - First array of the dot product
/// * `t_2` - Second array of the dot product
/// 
/// # Returns
/// * `Array2<f64>` - 2D Array containing the result of the dot product
pub fn par_dot(t_1: Array2<f64>, t_2: Array2<f64>) -> Array2<f64> {

    // Check for dimension compatibility
    let (m, n) = t_1.dim();
    let (n2, p) = t_2.dim();
    if n != n2 {
        panic!("Matrix dimensions do not match for multiplication: {}x{} and {}x{}, {} should be equal to {}", m, n, n2, p, n, n2);
    }

    // Empty array to store the data
    let mut a = Array2::zeros((m, p));

    // Parallel iteration over the rows of 'a' array
    a.axis_iter_mut(Axis(0))
    .into_par_iter()
    .enumerate()
    .for_each(|(i, mut row)| {
        for j in 0..p {
            row[j] = t_1.row(i).dot(&t_2.column(j)); //a row of t_1 is multiplied by a column of t_2 to get a row of the new array
        }
    });

    a
}