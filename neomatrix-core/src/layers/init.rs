use crate::tensor::Tensor;
use ndarray::Array2;
use ndarray_rand::{rand_distr::Normal, RandomExt};
use std::ops::Range;

pub enum Init {
    Random,
    Xavier,
    He,
}

impl Init {
    pub fn init(&self, in_feat: usize, out_feat: usize, rg: Option<Range<f32>>) -> Tensor {
        match self {
            Init::Random => Tensor::random(vec![in_feat, out_feat], rg.unwrap_or(-1.0..1.0)),
            Init::Xavier => {
                let std_dev = (2.0 / (in_feat + out_feat) as f32).sqrt();
                let dist = Normal::new(0.0, std_dev).unwrap();

                let data = Array2::random((in_feat, out_feat), dist).into_dyn();
                Tensor {
                    dimension: 2,
                    shape: vec![in_feat, out_feat],
                    data,
                }
            }
            Init::He => {
                let std_dev = (2.0 / (in_feat) as f32).sqrt();
                let dist = Normal::new(0.0, std_dev).unwrap();

                let data = Array2::random((in_feat, out_feat), dist).into_dyn();
                Tensor {
                    dimension: 2,
                    shape: vec![in_feat, out_feat],
                    data,
                }
            }
        }
    }
}
