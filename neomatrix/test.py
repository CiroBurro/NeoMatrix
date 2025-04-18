from rustybrain import Tensor


input = Tensor([4], [1, 2, 3, 4])
 
layer = Layer(3, input, Activation.Softmax)

z = layer.forward(parallel=False)

print(z)
print(z.data)

t = Tensor([3], [0.1, 0.2, 0.7])

error = get_cost(Cost.BinaryCrossEntropy, t, z, parallel=False, batch=False)
print(error)


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

batch_error = get_cost(Cost.MeanAbsoluteError, t, z, parallel=True, batch=True)
print(batch_error)
