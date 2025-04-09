from neomatrix import Tensor, Layer, Activation, Cost, get_cost

input = Tensor([4], [1, 2, 3, 4])

layer = Layer(3, input, Activation.Softmax)

z = layer.forward(parallel=False)

print(z)
print(z.data)

t = Tensor([3], [0.1, 0.2, 0.7])

batch_error = get_cost(Cost.HuberLoss, t, z, parallel=False, batch=False)


print(batch_error)
