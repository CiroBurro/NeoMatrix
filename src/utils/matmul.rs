/// This module provides a function for executing matrix multiplication in parallel using Rayon crate

/// Necessary imports
use ndarray::Array2;
use ndarray::Axis;
use ndarray::parallel::prelude::*;

pub fn par_dot(t_1: Array2<f64>, t_2: Array2<f64>) -> Array2<f64> {

    let (m, n) = t_1.dim();
    let (n2, p) = t_2.dim();

    if n != n2 {
        panic!("Le dimensioni delle matrici sono incompatibili per la moltiplicazione")
    }

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