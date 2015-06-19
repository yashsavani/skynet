#!/usr/bin/env python
import neocaffe
from neocaffe.layers import ConvLayer, DummyDataLayer, EuclideanLossLayer
import config
import h5py
import numpy as np

net = neocaffe.Net()
net.set_phase_train()
neocaffe.Caffe.set_random_seed(10)
neocaffe.Caffe.set_mode_cpu()
def iter(net):
    net.forward_layer(DummyDataLayer(name='image', shape=[config.batch_size, 1, 1, 1]))
    net.forward_layer(DummyDataLayer(name='label', shape=[config.batch_size, 1, 1, 1]))
    net.blobs['image'].data()[:] = 1
    net.blobs['label'].data()[0] = 3
    net.forward_layer(ConvLayer(name='conv', bottoms=['image'], kernel_h = 1, kernel_w = 1, num_output=1))
    net.forward_layer(EuclideanLossLayer(name='euclidean', bottoms=['conv', 'label']))
    #print 'euclidean loss: %s' % str(net.blobs['euclidean'].data())
    #print 'conv loss: %s' % str(net.blobs['conv'].diff())
    net.backward()
    net.update(lr=0.1, momentum=0.5, clip_gradients=.1)
    #print net.blobs['conv'].data()
    #print net.blobs['conv'].shape()
    print net.layer_params['conv'][0].data()

net.reshape_only = True
iter(net)
net.load('/tmp/a.h5')
net.reshape_only = False
for i in range(10):
    iter(net)

#net.save('/tmp/a.h5')
