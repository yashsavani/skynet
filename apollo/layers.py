import caffe_pb2
import apollo
from caffe_pb2 import DataParameter
import numpy as np

class Layer(object):
    def __init__(self, kwargs): #name, bottoms, tops, params=[]):
        name = kwargs['name']
        bottoms = kwargs.get('bottoms', [])
        tops = kwargs.get('tops', [name])
        param_names = kwargs.get('param_names', [])
        param_lr_mults = kwargs.get('param_lr_mults', [])
        assert type(tops) != str and type(bottoms) != str and type(param_names) != str
        self.p = caffe_pb2.LayerParameter()
        self.r = caffe_pb2.RuntimeParameter()
        self.p.name = name
        for blob_name in tops:
            self.p.top.append(blob_name)
        for blob_name in bottoms:
            self.p.bottom.append(blob_name)
        for i in range(max(len(param_names), len(param_lr_mults))):
            param = self.p.param.add()
            if param_names:
                param.name = param_names[i]
            if param_lr_mults:
                param.lr_mult = param_lr_mults[i]

class LossLayer(Layer):
    def __init__(self, kwargs):
        super(LossLayer ,self).__init__(kwargs)
        tops = kwargs.get('tops', [kwargs['name']])
        loss_weight = kwargs.get('loss_weight', 1.)
        self.p.loss_weight.append(loss_weight)
        assert len(tops) == 1

class Filler(object):
    def __init__(self, **kwargs):
        self.type = kwargs.get('type', None)
        self.value = kwargs.get('value', None)
        self.min = kwargs.get('min', None)
        self.max = kwargs.get('max', None)
        self.mean = kwargs.get('mean', None)
        self.std = kwargs.get('std', None)
        self.sparse = kwargs.get('sparse', None)
    def fill(self, param):
        if self.type is not None:
            param.type = self.type
        if self.value is not None:
            param.value = self.value
        if self.min is not None:
            param.min= self.min
        if self.max is not None:
            param.max = self.max
        if self.mean is not None:
            param.mean = self.mean
        if self.std is not None:
            param.std = self.std
        if self.sparse is not None:
            param.sparse = self.sparse

class Concat(Layer):
    def __init__(self, concat_dim=None, **kwargs):
        super(Concat, self).__init__(kwargs)
        self.p.type = type(self).__name__
        if concat_dim is not None:
            self.p.concat_param.concat_dim = concat_dim

class Convolution(Layer):
    def __init__(self, num_output, **kwargs):
        super(Convolution, self).__init__(kwargs)
        self.p.type = type(self).__name__
        self.p.convolution_param.num_output = num_output
        if kwargs.get('bias_term', None) is not None:
            self.p.convolution_param.bias_term = kwargs['bias_term']
        if kwargs.get('kernel_size', None) is not None:
            self.p.convolution_param.kernel_size = kwargs['kernel_size']
        if kwargs.get('kernel_h', None) is not None:
            self.p.convolution_param.kernel_h = kwargs['kernel_h']
        if kwargs.get('kernel_w', None) is not None:
            self.p.convolution_param.kernel_w = kwargs['kernel_w']
        if kwargs.get('pad', None) is not None:
            self.p.convolution_param.pad = kwargs['pad']
        if kwargs.get('pad_h', None) is not None:
            self.p.convolution_param.pad_h = kwargs['pad_h']
        if kwargs.get('pad_w', None) is not None:
            self.p.convolution_param.pad_w = kwargs['pad_w']
        if kwargs.get('stride', None) is not None:
            self.p.convolution_param.stride = kwargs['stride']
        if kwargs.get('stride_h', None) is not None:
            self.p.convolution_param.stride_h = kwargs['stride_h']
        if kwargs.get('stride_w', None) is not None:
            self.p.convolution_param.stride_w = kwargs['stride_w']
        if 'weight_filler' in kwargs:
            kwargs[weight_filler].fill(self.p.convolution_param.weight_filler)
        if 'bias_filler' in kwargs:
            kwargs[bias_filler].fill(self.p.convolution_param.bias_filler)

class Data(Layer):
    def __init__(self, source, batch_size, **kwargs):
        super(DummyData, self).__init__(kwargs)
        self.p.type = "Data"
        self.p.data_param.source = source
        self.p.data_param.backend = DataParameter.LMDB
        self.p.data_param.batch_size = batch_size

class Dropout(Layer):
    def __init__(self, dropout_ratio, **kwargs):
        super(Dropout, self).__init__(kwargs)
        self.p.type = type(self).__name__
        self.p.dropout_param.dropout_ratio = dropout_ratio

class DummyData(Layer):
    def __init__(self, shape, **kwargs):
        super(DummyData, self).__init__(kwargs)
        self.p.type = type(self).__name__
        assert len(shape) == 4
        self.p.dummy_data_param.num.append(shape[0])
        self.p.dummy_data_param.channels.append(shape[1])
        self.p.dummy_data_param.height.append(shape[2])
        self.p.dummy_data_param.width.append(shape[3])

class EuclideanLoss(LossLayer):
    def __init__(self, **kwargs):
        super(EuclideanLoss, self).__init__(kwargs)
        self.p.type = type(self).__name__

class InnerProduct(Layer):
    def __init__(self, num_output, bias_term=None, **kwargs):
        super(InnerProduct, self).__init__(kwargs)
        self.p.type = type(self).__name__
        self.p.inner_product_param.num_output = num_output
        if 'weight_filler' in kwargs:
            kwargs['weight_filler'].fill(self.p.wordvec_param.weight_filler)
        if bias_term is not None:
            self.p.inner_product_param.bias_term = bias_term

class Lstm(Layer):
    def __init__(self, num_cells, **kwargs):
        super(Lstm, self).__init__(kwargs)
        self.p.type = type(self).__name__
        self.p.lstm_param.num_cells = num_cells
        if 'weight_filler' in kwargs:
            kwargs['weight_filler'].fill(self.p.lstm_param.input_weight_filler)
            kwargs['weight_filler'].fill(self.p.lstm_param.input_gate_weight_filler)
            kwargs['weight_filler'].fill(self.p.lstm_param.forget_gate_weight_filler)
            kwargs['weight_filler'].fill(self.p.lstm_param.output_gate_weight_filler)

class NumpyData(Layer):
    def __init__(self, data, **kwargs):
        super(NumpyData, self).__init__(kwargs)
        self.p.type = type(self).__name__
        self.r = apollo.make_numpy_data_param(np.array(data, dtype=np.float32))
        # fast version of the following
        # for x in shape:
            # self.r.numpy_data_param.shape.append(x)
        # for x in data.flatten():
            # self.r.numpy_data_param.data.append(x)

class Pooling(Layer):
    def __init__(self, use_bias=False, **kwargs):
        super(Pooling, self).__init__(kwargs)
        self.p.type = type(self).__name__
        if kwargs.get('kernel_size', None) is not None:
            self.p.pooling_param.kernel_size = kwargs['kernel_size']
        if kwargs.get('kernel_h', None) is not None:
            self.p.pooling_param.kernel_h = kwargs['kernel_h']
        if kwargs.get('kernel_w', None) is not None:
            self.p.pooling_param.kernel_w = kwargs['kernel_w']
        if kwargs.get('pad', None) is not None:
            self.p.pooling_param.pad = kwargs['pad']
        if kwargs.get('pad_h', None) is not None:
            self.p.pooling_param.pad_h = kwargs['pad_h']
        if kwargs.get('pad_w', None) is not None:
            self.p.pooling_param.pad_w = kwargs['pad_w']
        if kwargs.get('stride', None) is not None:
            self.p.pooling_param.stride = kwargs['stride']
        if kwargs.get('stride_h', None) is not None:
            self.p.pooling_param.stride_h = kwargs['stride_h']
        if kwargs.get('stride_w', None) is not None:
            self.p.pooling_param.stride_w = kwargs['stride_w']
        if kwargs.get('global_pooling', None) is not None:
            self.p.pooling_param.global_pooling = kwargs['global_pooling']

class Relu(Layer):
    def __init__(self, **kwargs):
        super(Relu, self).__init__(kwargs)
        self.p.type = type(self).__name__

class Softmax(Layer):
    def __init__(self, ignore_label=None, normalize=None, **kwargs):
        super(Softmax, self).__init__(kwargs)
        self.p.type = type(self).__name__

class SoftmaxWithLoss(LossLayer):
    def __init__(self, ignore_label=None, normalize=None, **kwargs):
        super(SoftmaxWithLoss, self).__init__(kwargs)
        self.p.type = type(self).__name__
        if normalize is not None:
            self.p.loss_param.normalize = normalize
        if ignore_label is not None:
            self.p.loss_param.ignore_label = ignore_label

class Unknown(Layer):
    def __init__(self, p, r=None):
        self.p = p
        if r is None:
            self.r = caffe_pb2.RuntimeParameter()
        else:
            self.r = r

class Wordvec(Layer):
    def __init__(self, dimension, vocab_size, init_range, **kwargs):
        super(Wordvec, self).__init__(kwargs)
        self.p.type = type(self).__name__
        add_weight_filler(self.p.wordvec_param.weight_filler, 0.1)
        self.p.wordvec_param.dimension = dimension
        self.p.wordvec_param.vocab_size = vocab_size
        if 'weight_filler' in kwargs:
            kwargs['weight_filler'].fill(self.p.wordvec_param.weight_filler)
