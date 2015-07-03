#!/usr/bin/env python
import apollo
from apollo.layers import ConvLayer, DummyDataLayer, EuclideanLossLayer, LstmLayer, ConcatLayer, InnerProductLayer
import config
import h5py
import numpy as np
import random

net = apollo.Net()
net.set_phase_train()
apollo.Caffe.set_random_seed(10)
apollo.Caffe.set_mode_gpu()
def iter(net):
    net.forward_layer(DummyDataLayer(name='data', shape=[1, 1, 1, 1]))
    net.forward_layer(DummyDataLayer(name='label', shape=[1, 1, 1, 1]))
    value = random.random()
    net.blobs['data'].data()[0] = value 
    net.blobs['label'].data()[0] = value * 3
    net.forward_layer(ConvLayer(name='conv', bottoms=['data'], kernel_h = 1, kernel_w = 1, num_output=1))
    net.forward_layer(EuclideanLossLayer(name='euclidean', bottoms=['conv', 'label']))
    net.backward()
    net.update(lr=0.0001, momentum=0.9, clip_gradients=1)

error = 0
for i in range(10000000):
    iter(net)
    error += abs(net.blobs['conv'].data() - net.blobs['label'].data()) / 100
    if i % 100 == 0:
        print 'Iteration %d: %s' % (i, np.mean(error))
        error = 0

#net.save('/tmp/a.h5')
