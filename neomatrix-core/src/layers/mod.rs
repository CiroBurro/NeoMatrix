pub mod activations;
pub mod dense;
mod init;

use crate::errors::LayerError;
use crate::tensor::Tensor;

pub trait Layer {
    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor, LayerError>;
    fn backward(&mut self, output_gradient: &Tensor) -> Result<Tensor, LayerError>;
    fn get_params_and_grads(&mut self) -> Option<Vec<(&mut Tensor, &Tensor)>> {
        None
    }
}
