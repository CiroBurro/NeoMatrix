from neomatrix.core import Tensor, Layer, Activation, Cost, model

x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
training_x = Tensor([10, 2], x_data)

y_data = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
training_y = Tensor([10, 1], y_data)

input_nodes = 2
layer_1 = Layer(3, input_nodes, Activation.Linear)
out_layer = Layer(1, layer_1.nodes, Activation.Linear)

layers = [layer_1, out_layer]
nn = model.NeuralNetwork(layers, Cost.MeanSquaredError(), learning_rate=0.005)
nn.fit(training_set=training_x, training_targets=training_y, val_set=training_x, val_targets=training_y, epochs=100, batch_size=2, parallel=False)


t = Tensor([2], [4, 8])
output = nn.predict(ntwk_inputs=t, batch_processing=False, parallel=False)
print(t.data)
print(output.data)