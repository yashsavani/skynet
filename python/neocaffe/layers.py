import caffe_pb2
from caffe_pb2 import DataParameter


def add_weight_filler(param, max_value):
    param.type = 'uniform'
    param.min = -max_value
    param.max = max_value

class Layer(object):
    def __init__(self, kwargs): #name, bottoms, tops, params=[]):
        name = kwargs['name']
        bottoms = kwargs.get('bottoms', [])
        tops = kwargs.get('tops', [name])
        params = kwargs.get('params', [])
        assert type(tops) != str and type(bottoms) != str and type(params) != str
        self.p = caffe_pb2.LayerParameter()
        self.p.name = name
        for blob_name in tops:
            self.p.top.append(blob_name)
        for blob_name in bottoms:
            self.p.bottom.append(blob_name)
        for param_name in params:
            self.p.param.append(param_name)
         

class DummyDataLayer(Layer):
    def __init__(self, shape, **kwargs):
        super(DummyDataLayer, self).__init__(kwargs)
        self.p.type = "DummyData"
        assert len(shape) == 4
        self.p.dummy_data_param.num.append(shape[0])
        self.p.dummy_data_param.channels.append(shape[1])
        self.p.dummy_data_param.height.append(shape[2])
        self.p.dummy_data_param.width.append(shape[3])

class DataLayer(Layer):
    def __init__(self, source, batch_size, **kwargs):
        super(DummyDataLayer, self).__init__(kwargs)
        self.p.type = "Data"
        #self.p.data_param.source = 'examples/mnist/mnist_train_lmdb'
        self.p.data_param.source = source
        self.p.data_param.backend = DataParameter.LMDB
        self.p.data_param.batch_size = batch_size

class EuclideanLossLayer(Layer):
    def __init__(self, loss_weight=1., **kwargs):
        super(EuclideanLossLayer, self).__init__(kwargs)
        self.p.type = "EuclideanLoss"
        self.p.loss_weight.append(loss_weight)

class ConvLayer(Layer):
    def __init__(self, kernel_h, kernel_w, num_output, use_bias=False, **kwargs):
        super(ConvLayer, self).__init__(kwargs)
        self.p.type = "Convolution"
        self.p.convolution_param.bias_term = use_bias
        self.p.convolution_param.kernel_h = kernel_h
        self.p.convolution_param.kernel_w = kernel_w
        self.p.convolution_param.num_output = num_output
        add_weight_filler(self.p.convolution_param.weight_filler, 0.1)

class ReluLayer(Layer):
    def __init__(self, **kwargs):
        super(ReluLayer, self).__init__(kwargs)
        self.p.type = "ReLU"
