from neomatrix.core import Tensor, Layer, Activation, get_cost, Cost, model
from neomatrix.utils import get_batches
import os, time

input_nodes = 2
t_1 = Tensor([2], [1, 2])
t_2 = Tensor([2], [3, 4])
t_3 = Tensor([2], [5, 6])
t_4 = Tensor([2], [7, 8])
t_5 = Tensor([2], [9, 10])
t_6 = Tensor([2], [11, 12])
t_7 = Tensor([2], [13, 14])
t_8 = Tensor([2], [15, 16])
t_9 = Tensor([2], [17, 18])
t_10 = Tensor([2], [19, 20])

t_1.push_row(t_2)
t_1.push_row(t_3)
t_1.push_row(t_4)
t_1.push_row(t_5)
t_1.push_row(t_6)
t_1.push_row(t_7)
t_1.push_row(t_8)
t_1.push_row(t_9)
t_1.push_row(t_10)

training_set = t_1
#batches = get_batches(training_set, 1)
#print(batches)

t_1 = Tensor([2], [2, 1])
t_2 = Tensor([2], [4, 3])
t_3 = Tensor([2], [6, 5])
t_4 = Tensor([2], [8, 7])
t_5 = Tensor([2], [10, 9])
t_6 = Tensor([2], [12, 11])
t_7 = Tensor([2], [14, 13])
t_8 = Tensor([2], [16, 15])
t_9 = Tensor([2], [18, 17])
t_10 = Tensor([2], [20, 19])

t_1.push_row(t_2)
t_1.push_row(t_3)
t_1.push_row(t_4)
t_1.push_row(t_5)
t_1.push_row(t_6)
t_1.push_row(t_7)
t_1.push_row(t_8)
t_1.push_row(t_9)
t_1.push_row(t_10)

training_target = t_1


layer_1 = Layer(3, input_nodes, Activation.Relu)

layer_2 = Layer(4, layer_1.nodes, Activation.Relu)

layer_3 = Layer(5, layer_2.nodes, Activation.Tanh)

out_layer = Layer(2, layer_3.nodes, Activation.Softmax)


nn = model.NeuralNetwork(input_nodes, [layer_1, layer_2, layer_3, out_layer], Cost.MeanAbsoluteError, 0.01)

start_time = time.time()
nn.fit(training_set=training_set, training_targets=training_target, val_set=training_set, val_targets=training_target, epochs=50, batch_size=2, parallel=True)
stop_time = time.time()

time= stop_time - start_time
print(f"time: {time}")