from neomatrix import Layer, Tensor, Activation, get_cost, Cost

class NeuralNetwork:
    def __init__(self, inputs, layers):
        self.ntwk_inputs = inputs
        self.layers = layers

    def predict(self, parallel=False):
        inputs = self.ntwk_inputs
        for layer in self.layers:
            output = layer.forward(inputs, parallel=parallel)
            inputs = output
        return inputs



ntwk_input = Tensor([4], [1, 2, 3, 4])
layer_1 = Layer(5, ntwk_input.shape[0], Activation.Relu)
layer_2 = Layer(3, layer_1.nodes, Activation.Relu)
layer_3 = Layer(2, layer_2.nodes, Activation.Sigmoid)

nn = NeuralNetwork(ntwk_input, [layer_1, layer_2, layer_3])

z = nn.predict()
t = Tensor([2], [0.1, 0.9])

error = get_cost(Cost.MeanSquaredError, t, z, parallel=False, batch=False)

print(error)

deltas = layer_3.get_output_deltas(Cost.MeanSquaredError, t, z)
print(deltas)

print(layer_3.weights)

(w_grads, b_grads, out_deltas) = layer_3.backward(True, deltas, None, None)
print(w_grads)