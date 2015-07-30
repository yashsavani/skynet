#!/usr/bin/env python
import apollo
import imp
import numpy as np
import logging
import argparse

import caffe_pb2
from google.protobuf.text_format import Merge
from apollo import Net
import layers

class Architecture(object):
    """
    Class used when loading Caffe-style prototxts for legacy.py
    """
    def __init__(self, phase='train'):
        self.layers = []
        self.phase = phase
        self.phase_map = { 'train': caffe_pb2.TRAIN, 'test': caffe_pb2.TEST }
    def load_from_proto(self, prototxt):
        net = caffe_pb2.NetParameter()
        with open(prototxt, 'r') as f:
            Merge(f.read(), net)
        def xor(a, b):
            return (a and (not b)) or ((not a) and b)
        assert xor(len(net.layer) > 0, len(net.layers) > 0), \
            "Net cannot have both new and old layer types."
        if net.layer:
            net_layers = net.layer
        else:
            net_layers = net.layers
        for layer in net_layers:
            include_phase_list = map(lambda x: x.phase, layer.include)
            if len(include_phase_list) > 0 and self.phase_map[self.phase] not in include_phase_list:
                continue
            new_layer = layers.Unknown({})
            new_layer.p = layer
            if len(new_layer.p.top) == 0:
                if str(layer.type) != "Silence":
                    # hack to handle Silence layer not having a top
                    new_layer.p.top.append(new_layer.p.name)
            self.layers.append(new_layer)
        return net
    def forward(self, net):
        loss = 0
        for x in self.layers:
            loss += net.forward_layer(x)
        return loss

def run(hyper):
    net = apollo.Net()
    arch = Architecture()
    arch.load_from_proto(hyper['net_prototxt'])

    test_net = apollo.Net(phase='test')
    test_arch = Architecture(phase='test')
    test_arch.load_from_proto(hyper['net_prototxt'])

    arch.forward(net)
    if hyper['weights']:
        print 'loading weights from %s' % hyper['weights']
        net.load(hyper['weights'])
    net.reset_forward()

    test_arch.forward(test_net)
    test_net.reset_forward()

    def test_performance(net, test_net):
        test_error = []
        test_net.copy_params_from(net)
        for _ in range(hyper['test_iter']):
            test_error.append(test_arch.forward(test_net))
            test_net.reset_forward()
        logging.info('Test Error: %f' % np.mean(test_error))

    error = []
    for i in range(hyper['max_iter']):
        error.append(arch.forward(net))
        net.backward()
        lr = (hyper['base_lr'] / (hyper.get('gamma', 1.))**(i // hyper['stepsize']))
        net.update(lr=lr, momentum=hyper['momentum'],
            clip_gradients=hyper.get('clip_gradients', -1), weight_decay=hyper['weight_decay'])
        if i % hyper['display_interval'] == 0:
            logging.info('Iteration %d: %s' % (i, np.mean(error)))
            error = []
        if i % hyper['test_interval'] == 0:
            test_performance(net, test_net)
            if 'accuracy' in test_net.tops:
                logging.info('Accuracy: %.5f' % test_net.tops['accuracy'].data.flatten()[0])
        if i % hyper['snapshot_interval'] == 0 and i > 0:
            net.save('%s_%d.h5' % (hyper['snapshot_prefix'], i))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', default=3, type=int)
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--solver')
    args = parser.parse_args()

    config = imp.load_source('module.name', args.solver)
    hyper = config.get_hyper()
    hyper['weights'] = args.weights

    if args.gpu is not None:
        apollo.Caffe.set_mode_gpu()
        apollo.Caffe.set_device(int(args.gpu))
    else:
        apollo.Caffe.set_mode_cpu()
    apollo.Caffe.set_random_seed(hyper.get('random_seed', 0))
    apollo.Caffe.set_logging_verbosity(args.loglevel)
    run(hyper)


if __name__ == '__main__':
    main()
