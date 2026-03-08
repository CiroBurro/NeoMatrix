use crate::{
    errors::LayerError,
    layers::Layer,
    math::activations::{self, ActivationFunction},
    tensor::Tensor,
};

pub struct ReLu {
    inner: activations::Relu,
    input_cache: Option<Tensor>,
}
impl Layer for ReLu {
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError> {
        if training {
            self.input_cache = Some(input.clone());
        }

        self.inner.function(input).map_err(LayerError::from)
    }

    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError> {
        output_gradient
            .dot(
                &self
                    .inner
                    .derivative(
                        self.input_cache
                            .as_ref()
                            .ok_or(LayerError::NotInitialized)?,
                    )
                    .map_err(LayerError::from)?,
            )
            .map_err(LayerError::from)
    }
}

pub struct Sigmoid {
    inner: activations::Sigmoid,
    output_cache: Option<Tensor>,
}
impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError> {
        let output = self.inner.function(input).map_err(LayerError::from);

        if training {
            self.output_cache = Some(output?);
        }

        self.inner.function(input).map_err(LayerError::from)
    }

    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError> {
        let prev_output = self
            .output_cache
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        (output_gradient * ((prev_output * (1.0 - prev_output)).map_err(LayerError::from)?))
            .map_err(LayerError::from)
    }
}

pub struct Tanh {
    inner: activations::Tanh,
    output_cache: Option<Tensor>,
}
impl Layer for Tanh {
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError> {
        let output = self.inner.function(input).map_err(LayerError::from);

        if training {
            self.output_cache = Some(output?);
        }

        self.inner.function(input).map_err(LayerError::from)
    }

    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError> {
        let prev_output = self
            .output_cache
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        (output_gradient * (1.0 - (prev_output * prev_output).map_err(LayerError::from)?))
            .map_err(LayerError::from)
    }
}

pub struct Softmax {
    inner: activations::Softmax,
    input_cache: Option<Tensor>,
}
impl Layer for Softmax {
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError> {
        if training {
            self.input_cache = Some(input.clone());
        }

        self.inner.function(input).map_err(LayerError::from)
    }

    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError> {
        Ok(output_gradient.clone())
    }
}
