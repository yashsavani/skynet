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
                    new_layer.p.top.append(new_layer.p.name)
            self.layers.append(new_layer)
        return net
    def forward(self, net, hyper):
        loss = 0
        for x in self.layers:
            loss += net.forward_layer(x)
        return loss

def main():
    parser = apollo.utils.training.default_parser()
    parser.add_argument('--solver')
    args = parser.parse_args()

    config = imp.load_source('module.name', args.solver)
    hyper = {}
    hyper.update(config.get_hyper())
    hyper.update({k:v for k, v in vars(args).iteritems() if v is not None})

    arch = Architecture()
    arch.load_from_proto(hyper['net_prototxt'])

    test_net = apollo.Net(phase='test')
    test_arch = Architecture(phase='test')
    test_arch.load_from_proto(hyper['net_prototxt'])

    apollo.utils.training.default_train(hyper, forward=arch.forward, test_forward=test_arch.forward)


if __name__ == '__main__':
    main()
