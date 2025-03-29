from neomatrix import Tensor, Layer, Activation
import time

start = time.time()
t = Tensor([20], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

layer = Layer(3, t, Activation.Softmax)

output = layer.forward(False)
end = time.time() - start
print(output.data)
print("time: " + str(end))

start = time.time() - end
t = Tensor.zeros([20])

layer = Layer(3, t, Activation.Softmax)

output = layer.forward(True)
end = time.time() - start
print(output.data)
print("time: " + str(end))
