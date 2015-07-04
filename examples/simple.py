import apollo
from apollo import layers
import numpy as np

net = apollo.Net()
for i in range(1000):
    example = np.array(np.random.random()).reshape((1,1,1,1))
    net.forward_layer(layers.NumpyData(name='data', data=example))
    net.forward_layer(layers.NumpyData(name='label', data=(example*3)))
    net.forward_layer(layers.Convolution(name='conv', kernel_size=1, bottoms=['data'], num_output=1))
    loss = net.forward_layer(layers.EuclideanLoss(name='loss', bottoms=['conv', 'label']))
    net.backward()
    net.update(lr=0.1)
    if i % 100 == 0:
        print loss
