from neomatrix import Tensor, Layer, Activation, Cost, get_cost, random_weights, random_biases

input = Tensor([4], [1, 2, 3, 4])

layer = Layer(3, input, Activation.Softmax)

z = layer.forward(parallel=False)

print(z)
print(z.data)

t = Tensor([3], [0.1, 0.2, 0.7])

batch_error = get_cost(Cost.HuberLoss, t, z, parallel=False, batch=False)


print(batch_error)


weights = random_biases(layer.nodes, (-0.5, 0.5))
print(weights)
print(weights.data)


# concatenare due tensori
t_1 = Tensor([3], [1, 2, 3])
t_2 = Tensor([3], [1, 2, 4])
t_3 = Tensor([3], [1, 2, 5])
t_4 = Tensor([3], [1, 2, 6])
t_5 = Tensor([3], [1, 2, 7])

t = Tensor([5, 3], [*t_1.data, *t_2.data, *t_3.data, *t_4.data, *t_5.data])

z_1 = Tensor([3], [1, 2, 2])
z_2 = Tensor([3], [1, 2, 3])
z_3 = Tensor([3], [1, 2, 4])
z_4 = Tensor([3], [1, 2, 5])
z_5 = Tensor([3], [1, 2, 6])

z = Tensor([5, 3], [*z_1.data, *z_2.data, *z_3.data, *z_4.data, *z_5.data])