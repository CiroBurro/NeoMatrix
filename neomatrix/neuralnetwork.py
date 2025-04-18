from rustybrain import Tensor, Layer, Activation, get_cost, Cost

class NeuralNetwork:
    def __init__(self, inputs: Tensor, layers: [Layer], cost_function: Cost):
        self.ntwk_inputs = inputs
        self.layers = layers
        self.cost_function = cost_function

    def predict(self, parallel=False):
        inputs = self.ntwk_inputs
        for layer in self.layers:
            output = layer.forward(inputs, parallel=parallel)
            inputs = output
        return inputs

    def backward(self, t: Tensor, z: Tensor, batch=True):
        _error = get_cost(self.cost_function, t, z, parallel=False, batch=batch)
        deltas = self.layers[-1].get_output_deltas(self.cost_function, t, z)
        for (i, layer) in enumerate(reversed(self.layers)):
            out_layer = True
            next_weights = None
            if i != 0:
                out_layer = False
                next_weights = self.layers[-(i + 1)].weights

            (w_grads, b_grads, deltas) = layer.backward(out_layer, deltas, next_weights)


ntwk_input = Tensor([4], [1, 2, 3, 4])
layer_1 = Layer(5, ntwk_input.shape[0], Activation.Relu)
layer_2 = Layer(3, layer_1.nodes, Activation.Relu)
layer_3 = Layer(2, layer_2.nodes, Activation.Sigmoid)

nn = NeuralNetwork(ntwk_input, [layer_1, layer_2, layer_3], Cost.MeanSquaredError)

z = nn.predict()
t = Tensor([2], [0.1, 0.9])

error = get_cost(Cost.MeanSquaredError, t, z, parallel=False, batch=False)


deltas = layer_3.get_output_deltas(nn.cost_function, t, z)
print("Deltas:")
print(deltas)

(w_grads, b_grads, out_deltas) = layer_3.backward(True, deltas, None, None)
print("\nLayer 3:")
print(layer_3.weights)
print(w_grads)

print("\nOut deltas:")
print(out_deltas)

(w2_grads, b2_grads, deltas2) = layer_2.backward(False, out_deltas, layer_3.weights, None)
print("\nLayer 2:")
print(layer_2.weights)
print(w2_grads)

(w1_grads, b1_grads, deltas1) = layer_1.backward(False, deltas2, layer_2.weights, None)
print("\nLayer 1:")
print(layer_1.weights)
print(w1_grads)

