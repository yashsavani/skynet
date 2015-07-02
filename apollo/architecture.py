import caffe_pb2
from google.protobuf.text_format import Merge
from ._apollo import Net
import layers

class Architecture(object):
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
            self.layers.append(new_layer)
        return net
    def forward(self, net):
        loss = 0
        for x in self.layers:
            loss += net.forward_layer(x)
        return loss
