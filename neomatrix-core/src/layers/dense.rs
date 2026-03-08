use std::ops::Range;

use ndarray::Axis;

use crate::errors::LayerError;
use crate::layers::init::Init;
use crate::layers::Layer;
use crate::tensor::Tensor;

pub struct Dense {
    input_cache: Option<Tensor>,
    weights: Tensor,
    biases: Tensor,
    weights_gradient: Option<Tensor>,
    biases_gradient: Option<Tensor>,
}

impl Dense {
    pub fn new(
        in_feat: usize,
        out_feat: usize,
        init: Option<Init>,
        rg: Option<Range<f32>>,
    ) -> Self {
        Self {
            input_cache: None,
            weights: init
                .unwrap_or(Init::Xavier)
                .init(in_feat, out_feat, rg.clone()),
            biases: Tensor::zeros(vec![out_feat]),
            weights_gradient: None,
            biases_gradient: None,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError> {
        if training {
            self.input_cache = Some(input.clone());
        }

        (&input.dot(&self.weights)? + &self.biases).map_err(LayerError::from)
    }

    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .input_cache
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        self.weights_gradient = Some(input.transpose()?.dot(output_gradient)?);

        let data = output_gradient.data.sum_axis(Axis(0)).into_dyn();
        self.biases_gradient = Some(Tensor {
            dimension: data.ndim(),
            shape: data.shape().to_vec(),
            data,
        });

        output_gradient
            .dot(&self.weights.transpose()?)
            .map_err(LayerError::from)
    }

    fn get_params_and_grads(&mut self) -> Option<Vec<(&mut Tensor, &Tensor)>> {
        if self.weights_gradient.is_some() && self.biases_gradient.is_some() {
            Some(vec![
                (&mut self.weights, self.weights_gradient.as_ref().unwrap()),
                (&mut self.biases, self.biases_gradient.as_ref().unwrap()),
            ])
        } else {
            None
        }
    }
}
