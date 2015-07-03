#!/usr/bin/env python
import apollo
import config
import numpy as np
import logging

hyper = config.get_hyper()

apollo.Caffe.set_random_seed(hyper.random_seed)
apollo.Caffe.set_mode_gpu()
apollo.Caffe.set_device(0)
apollo.Caffe.set_logging_verbosity(3)

net = apollo.Net()
arch = apollo.Architecture()
arch.load_from_proto(hyper.net)

test_net = apollo.Net(phase='test')
test_arch = apollo.Architecture(phase='test')
test_arch.load_from_proto(hyper.net)

net.forward(arch)
net.load('../nlpcaffe/examples/language_model/lm_iter_20000.caffemodel')
net.reset_forward()

test_arch.forward(test_net)
test_net.reset_forward()

def test_performance(net, test_net):
    test_error = []
    test_net.copy_params_from(net)
    for _ in range(hyper.test_iter):
        test_error.append(test_net.forward(test_arch))
        test_net.reset_forward()
    logging.info('Test Error: %f' % np.mean(test_error))

error = []
for i in range(hyper.max_iter):
    error.append(net.forward(arch))
    net.backward()
    lr = (hyper.lr / (hyper.gamma)**(i // hyper.stepsize))
    net.update(lr=lr, momentum=hyper.momentum,
        clip_gradients=hyper.clip_gradients, decay_rate=hyper.decay_rate)
    if i % hyper.display_interval == 0:
        logging.info('Iteration %d: %s' % (i, np.mean(error)))
        error = []
    if i % hyper.test_interval == 0:
        test_performance(net, test_net)
    if i % hyper.snapshot_interval == 0 and i > 0:
        net.save('%s_%d.h5' % (hyper.snapshot_prefix, i))
