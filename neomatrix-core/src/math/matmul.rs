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
