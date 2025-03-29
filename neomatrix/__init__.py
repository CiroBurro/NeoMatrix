from neomatrix import Tensor, Layer, Activation

ntkw_input = Tensor([5], [1, 2, 3, 4, 5])

hidden_layer_1 = Layer(4, ntkw_input, Activation.Relu)
output_1 = hidden_layer_1.forward()
print(output_1)
print(output_1.data)

hidden_layer_2 = Layer(6, output_1, Activation.Relu)
output_2 = hidden_layer_2.forward()
print(hidden_layer_2.weights)
print(hidden_layer_2.weights.data)

output_layer = Layer(2, output_2, Activation.Sigmoid)
output = output_layer.forward()
print(output)
print(output.data)