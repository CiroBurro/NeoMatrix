import neomatrix.core as core
import neomatrix.utils as utils
import os

__all__ = [
    'NeuralNetwork',
    'LinearRegression',
    'LogisticRegression',
    'SoftmaxRegression',
]

class NeuralNetwork:
    def __init__(self, layers: list[core.Layer], cost_function: core.Cost, learning_rate: float):
        self.layers = layers
        self.cost_function = cost_function
        self.learning_rate = learning_rate

    def predict(self, ntwk_inputs: core.Tensor, batch_processing: bool=True, parallel: bool=False) -> core.Tensor:
        inputs = ntwk_inputs
        if batch_processing:
            for layer in self.layers:
                output = layer.forward_batch(inputs, parallel=parallel)
                inputs = output
            return inputs
        else:
            if inputs.dimension == 2:
                exit(1)
            for layer in self.layers:
                output = layer.forward(inputs, parallel=parallel)
                inputs = output
            return inputs

    def backward(self, ntwk_inputs: core.Tensor, t: core.Tensor, z: core.Tensor):
        deltas = self.layers[-1].get_output_deltas(self.cost_function, t, z)
        for (i, layer) in enumerate(reversed(self.layers)):
            out_layer = True
            next_weights = None
            if i != 0:
                out_layer = False
                next_weights = self.layers[-i].weights
            
            if i + 1 < len(self.layers):
                prev_layer = self.layers[- (i + 2)]
                all_outputs = prev_layer.output
            elif i + 1 == len(self.layers):
                all_outputs = ntwk_inputs
                 
            (w_grads, b_grads, new_deltas) = layer.backward(out_layer, deltas, next_weights, all_outputs)
            layer.weights = layer.weights.tensor_subtraction(w_grads.scalar_multiplication(self.learning_rate))
            layer.biases = layer.biases.tensor_subtraction(b_grads.scalar_multiplication(self.learning_rate))
            deltas = new_deltas
    
    def fit(self, training_set: core.Tensor, training_targets: core.Tensor, val_set: core.Tensor, val_targets: core.Tensor, epochs: int, batch_size: int, parallel: bool = False):
        batch_processing: bool = True
        if batch_size == 1 or batch_size == 0:
            batch_size = 1
            batch_processing = False
        
        
        for (i, epoch) in enumerate(range(epochs)):
            os.system('cls' if os.name == 'nt' else 'clear')
            total_loss = 0
            train_batches = utils.get_batches(training_set, batch_size)
            train_target_batches = utils.get_batches(training_targets, batch_size)
            val_batches = utils.get_batches(val_set, batch_size)
            val_targets_batches = utils.get_batches(val_targets, batch_size)

            for (j, batch) in enumerate(train_batches):
                outputs = self.predict(ntwk_inputs=batch, batch_processing=batch_processing, parallel=parallel)
                self.backward(ntwk_inputs=batch, t=train_target_batches[j], z=outputs)
                loss = core.get_cost(self.cost_function, train_target_batches[j], z=outputs, parallel=parallel, batch_processing=batch_processing)
                total_loss += loss
                print("-------------------------")
                print(f"Epoch: {i + 1}, Training batch: {j}, Loss: {loss}, Total loss: {total_loss}")
            
            for (k, batch) in enumerate(val_batches):
                outputs = self.predict(ntwk_inputs=batch, batch_processing=batch_processing, parallel=parallel)
                val_loss = core.get_cost(self.cost_function, t=val_targets_batches[k], z=outputs, parallel=parallel, batch_processing=batch_processing)
                total_loss += val_loss
                print("-------------------------")
                print(f"Epoch: {i + 1}, Validation batch: {j}, Loss: {val_loss}, Total loss: {total_loss}")
            
class LinearRegression(NeuralNetwork):
    def __init__(self, input_nodes: int, output_nodes: int, learning_rate: float):
        super().__init__([
            core.Layer(output_nodes, input_nodes, core.Activation.Linear)
        ], core.Cost.MeanSquaredError(), learning_rate)

class LogisticRegression(NeuralNetwork):
    def __init__(self, input_nodes: int, learning_rate: float):
        super().__init__([
            core.Layer(1, input_nodes, core.Activation.Sigmoid)
        ], core.Cost.BinaryCrossEntropy(), learning_rate)

class SoftmaxRegression(NeuralNetwork):
    def __init__(self, input_nodes: int, output_nodes: int, learning_rate: float):
        super().__init__([
            core.Layer(output_nodes, input_nodes, core.Activation.Softmax)
        ], core.Cost.CategoricalCrossEntropy(), learning_rate)
