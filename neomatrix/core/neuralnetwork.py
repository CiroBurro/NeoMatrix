from . import Tensor, Layer, Activation, get_cost, Cost
from neomatrix.utils.dataset import get_batches
import numpy as np
import os

class NeuralNetwork:
    def __init__(self, input_nodes: int, layers: [Layer], cost_function: Cost):
        self.input_nodes = input_nodes
        self.layers = layers
        self.cost_function = cost_function

    def predict(self, ntwk_inputs: Tensor, batch_processing: bool=True, parallel: bool=False) -> Tensor:
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

    def backward(self, ntwk_inputs: Tensor, t: Tensor, z: Tensor, batch_processing: bool, parallel: bool=False):
        error = get_cost(self.cost_function, t, z, parallel, batch_processing)
        deltas = self.layers[-1].get_output_deltas(self.cost_function, t, z)
        for (i, layer) in enumerate(reversed(self.layers)):
            out_layer = True
            next_weights = None
            if i != 0:
                out_layer = False
                next_weights = self.layers[-i].weights
            
            if i + 1 < len(self.layers):
                prev_layer = self.layers[i + 1]
                all_outputs = prev_layer.output
            elif i + 1 == len(self.layers):
                all_outputs = ntwk_inputs
                 
            (w_grads, b_grads, new_deltas) = layer.backward(out_layer, deltas, next_weights, all_outputs)
            deltas = new_deltas

        return error
    
    def train(self, training_set: Tensor, training_targets: Tensor, val_set: Tensor, val_targets: Tensor, epochs: int, batch_size: int, parallel: bool = False):
        batch_processing: bool = True
        if batch_size == 0 or batch_size == 1:
            batch_processing = False
        
        for (i, epoch) in enumerate(range(epochs)):
            total_loss = 0
            if batch_processing:
                train_batches = get_batches(training_set, batch_size)
                train_target_batches = get_batches(training_targets, batch_size)
                val_batches = get_batches(val_set, batch_size)
                val_targets_batches = get_batches(val_targets, batch_size)

                for (j, batch) in enumerate(train_batches):
                    outputs = self.predict(ntwk_inputs=batch, batch_processing=batch_processing, parallel=parallel)
                    loss = self.backward(ntwk_inputs=batch, t=train_target_batches[j], z=outputs, batch_processing=batch_processing, parallel=parallel)
                    total_loss += loss
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("-------------------------\n")
                    print(f"Epoch: {i}, Training batch: {j}, Loss: {loss}, Total loss: {total_loss}")
                
                for (k, batch) in enumerate(val_batches):
                    outputs = self.predict(ntwk_inputs=batch, batch_processing=batch_processing, parallel=parallel)
                    val_loss = get_cost(self.cost_function, val_targets_batches, outputs, parallel=parallel, batch_processing=batch_processing)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("-------------------------\n")
                    print(f"Epoch: {i}, Validation batch: {j}, Loss: {val_loss}")
            else:
                pass
                
            