use crate::structures::tensor::Tensor;
use ndarray::parallel::prelude::*;
use ndarray::{s, Axis};
use pyo3::prelude::*;

pub trait CostFunction: Send + Sync {
    fn function(&self, t: Tensor, z: Tensor) -> f64;
    fn function_batch(&self, t: Tensor, z: Tensor) -> f64;
    fn par_function_batch(&self, t: Tensor, z: Tensor) -> f64;
    fn derivative(&self, t: Tensor, z: Tensor) -> Tensor;
}

#[pyclass]
#[derive(Clone, Debug)]
pub enum Cost {
    MeanSquaredError,
    MeanAbsoluteError,
    BinaryCrossEntropy,
    HuberLoss,
    HingeLoss,
}

#[pyfunction]
pub fn get_cost(
    cost: Cost,
    t: Tensor,
    z: Tensor,
    parallel: Option<bool>,
    batch: Option<bool>,
) -> f64 {
    let parallel = parallel.unwrap_or(false);
    let batch = batch.unwrap_or(true);
    let f: Box<dyn CostFunction> = match cost {
        Cost::MeanSquaredError => Box::new(MeanSquaredError),
        Cost::MeanAbsoluteError => Box::new(MeanAbsoluteError),
        Cost::BinaryCrossEntropy => Box::new(BinaryCrossEntropy),
        Cost::HuberLoss => Box::new(HuberLoss),
        Cost::HingeLoss => Box::new(HingeLoss),
    };
    if !batch {
        f.function(t, z)
    } else {
        if parallel {
            f.par_function_batch(t, z)
        } else {
            f.function_batch(t, z)
        }
    }
}

pub struct MeanSquaredError;
impl CostFunction for MeanSquaredError {
    fn function(&self, t: Tensor, z: Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 1 {
            panic!("Dimensioni dei vettori output incompatibii")
        }
        let nomin = t
            .subtract(&z)
            .expect("Sottrazione tra i vettori non riuscita")
            .data
            .mapv(|x| x.powi(2))
            .sum();
        let denom = t.shape[0] as f64;
        nomin / denom
    }
    fn function_batch(&self, t: Tensor, z: Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 2 {
            panic!("Dimensioni dei vettori incompatibii")
        }

        let (m, n) = (t.shape[0], t.shape[1]);
        let sum: f64 = t
            .data
            .axis_iter(Axis(0))
            .zip(z.data.axis_iter(Axis(0)))
            .map(|(t_i, z_i)| {
                let t_i = Tensor {
                    dimension: 1,
                    shape: vec![t.shape[1]],
                    data: t_i.to_owned(),
                };
                let z_i = Tensor {
                    dimension: 1,
                    shape: vec![z.shape[1]],
                    data: z_i.to_owned(),
                };
                self.function(t_i, z_i)
            })
            .sum();
        sum / m as f64
    }
    fn par_function_batch(&self, t: Tensor, z: Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 2 {
            panic!("Dimensioni dei vettori incompatibii")
        }

        let (m, n) = (t.shape[0], t.shape[1]);
        let sum: f64 = t
            .data
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(z.data.axis_iter(Axis(0)).into_par_iter())
            .map(|(t_i, z_i)| {
                let t_i = Tensor {
                    dimension: 1,
                    shape: vec![t.shape[1]],
                    data: t_i.to_owned(),
                };
                let z_i = Tensor {
                    dimension: 1,
                    shape: vec![z.shape[1]],
                    data: z_i.to_owned(),
                };
                self.function(t_i, z_i)
            })
            .sum();
        sum / m as f64
    }
    fn derivative(&self, t: Tensor, z: Tensor) -> Tensor {
        todo!()
    }
}
pub struct MeanAbsoluteError;
impl CostFunction for MeanAbsoluteError {
    fn function(&self, t: Tensor, z: Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 1 {
            panic!("Dimensioni dei vettori output incompatibii")
        }
        let nomin = t
            .subtract(&z)
            .expect("Sottrazione tra i vettori non riuscita")
            .data
            .mapv(|x| x.abs())
            .sum();
        let denom = t.shape[0] as f64;
        nomin / denom
    }
    fn function_batch(&self, t: Tensor, z: Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 2 {
            panic!("Dimensioni dei vettori incompatibii")
        }

        let (m, n) = (t.shape[0], t.shape[1]);
        let sum: f64 = t
            .data
            .axis_iter(Axis(0))
            .zip(z.data.axis_iter(Axis(0)))
            .map(|(t_i, z_i)| {
                let t_i = Tensor {
                    dimension: 1,
                    shape: vec![t.shape[1]],
                    data: t_i.to_owned(),
                };
                let z_i = Tensor {
                    dimension: 1,
                    shape: vec![z.shape[1]],
                    data: z_i.to_owned(),
                };
                self.function(t_i, z_i)
            })
            .sum();
        sum / m as f64
    }
    fn par_function_batch(&self, t: Tensor, z: Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 2 {
            panic!("Dimensioni dei vettori incompatibii")
        }

        let (m, n) = (t.shape[0], t.shape[1]);
        let sum: f64 = t
            .data
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(z.data.axis_iter(Axis(0)).into_par_iter())
            .map(|(t_i, z_i)| {
                let t_i = Tensor {
                    dimension: 1,
                    shape: vec![t.shape[1]],
                    data: t_i.to_owned(),
                };
                let z_i = Tensor {
                    dimension: 1,
                    shape: vec![z.shape[1]],
                    data: z_i.to_owned(),
                };
                self.function(t_i, z_i)
            })
            .sum();
        sum / m as f64
    }
    fn derivative(&self, t: Tensor, z: Tensor) -> Tensor {
        todo!()
    }
}
pub struct BinaryCrossEntropy;
impl CostFunction for BinaryCrossEntropy {
    fn function(&self, t: Tensor, z: Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 1 {
            panic!("Dimensioni dei vettori output incompatibii")
        }

        let sum = t
            .data
            .to_owned()
            .iter()
            .zip(z.data.to_owned().iter())
            .map(|(t_i, z_i)| t_i * z_i.ln() + (1 - t_i) * (1.0 - z_i).ln())
            .sum();

        sum * (-1.0) / t.shape[0] as f64
    }
    fn function_batch(&self, t: Tensor, z: Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 2 {
            panic!("Dimensioni dei vettori output incompatibii")
        }

        let (m, n) = (t.shape[0], t.shape[1]);

        let sum = t
            .data
            .axis_iter(Axis(0))
            .zip(z.data.axis_iter(Axis(0)))
            .map(|(t_i, z_i)| {
                let t_i = Tensor {
                    dimension: 1,
                    shape: vec![t.shape[1]],
                    data: t_i.to_owned(),
                };
                let z_i = Tensor {
                    dimension: 1,
                    shape: vec![z.shape[1]],
                    data: z_i.to_owned(),
                };
                self.function(t_i, z_i)
            })
            .sum();

        sum / m as f64
    }
    fn par_function_batch(&self, t: Tensor, z: Tensor) -> f64 {
        if t.shape != z.shape || t.dimension != 2 {
            panic!("Dimensioni dei vettori output incompatibii")
        }

        let (m, n) = (t.shape[0], t.shape[1]);

        let sum = t
            .data
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(z.data.axis_iter(Axis(0)).into_par_iter())
            .map(|(t_i, z_i)| {
                let t_i = Tensor {
                    dimension: 1,
                    shape: vec![t.shape[1]],
                    data: t_i.to_owned(),
                };
                let z_i = Tensor {
                    dimension: 1,
                    shape: vec![z.shape[1]],
                    data: z_i.to_owned(),
                };
                self.function(t_i, z_i)
            })
            .sum();

        sum / m as f64
    }
    fn derivative(&self, t: Tensor, z: Tensor) -> Tensor {
        todo!()
    }
}
pub struct HuberLoss;
impl CostFunction for HuberLoss {
    fn function(&self, t: Tensor, z: Tensor) -> f64 {
        todo!()
    }
    fn function_batch(&self, t: Tensor, z: Tensor) -> f64 {
        todo!()
    }
    fn par_function_batch(&self, t: Tensor, z: Tensor) -> f64 {
        todo!()
    }
    fn derivative(&self, t: Tensor, z: Tensor) -> Tensor {
        todo!()
    }
}
pub struct HingeLoss;
impl CostFunction for HingeLoss {
    fn function(&self, t: Tensor, z: Tensor) -> f64 {
        todo!()
    }
    fn function_batch(&self, t: Tensor, z: Tensor) -> f64 {
        todo!()
    }
    fn par_function_batch(&self, t: Tensor, z: Tensor) -> f64 {
        todo!()
    }
    fn derivative(&self, t: Tensor, z: Tensor) -> Tensor {
        todo!()
    }
}
