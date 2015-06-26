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
        param_names = kwargs.get('param_names', [])
        param_lr_mults = kwargs.get('param_lr_mults', [])
        force_reshape = kwargs.get('force_reshape', None)
        assert type(tops) != str and type(bottoms) != str and type(param_names) != str
        self.p = caffe_pb2.LayerParameter()
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
        if force_reshape is not None:
            self.p.force_reshape = True
         

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

class LstmLayer(Layer):
    def __init__(self, num_cells, init_range, **kwargs):
        super(LstmLayer, self).__init__(kwargs)
        self.p.type = "Lstm"
        self.p.lstm_param.num_cells = num_cells
        add_weight_filler(self.p.lstm_param.input_weight_filler, init_range)
        add_weight_filler(self.p.lstm_param.input_gate_weight_filler, init_range)
        add_weight_filler(self.p.lstm_param.forget_gate_weight_filler, init_range)
        add_weight_filler(self.p.lstm_param.output_gate_weight_filler, init_range)

class InnerProductLayer(Layer):
    def __init__(self, num_output, init_range, bias_term=None, **kwargs):
        super(InnerProductLayer, self).__init__(kwargs)
        self.p.type = "InnerProduct"
        self.p.inner_product_param.num_output = num_output
        add_weight_filler(self.p.lstm_param.output_gate_weight_filler, init_range)
        if bias_term is not None:
            self.p.inner_product_param.bias_term = bias_term

class ConcatLayer(Layer):
    def __init__(self, concat_dim=None, **kwargs):
        super(ConcatLayer, self).__init__(kwargs)
        self.p.type = "Concat"
        if concat_dim is not None:
            self.p.concat_param.concat_dim = concat_dim

class ConvLayer(Layer):
    def __init__(self, kernel_h, kernel_w, num_output, use_bias=False, **kwargs):
        super(ConvLayer, self).__init__(kwargs)
        self.p.type = "Convolution"
        self.p.convolution_param.bias_term = use_bias
        self.p.convolution_param.kernel_h = kernel_h
        self.p.convolution_param.kernel_w = kernel_w
        self.p.convolution_param.num_output = num_output
        add_weight_filler(self.p.convolution_param.weight_filler, 0.1)

class WordvecLayer(Layer):
    def __init__(self, dimension, vocab_size, init_range, **kwargs):
        super(WordvecLayer, self).__init__(kwargs)
        self.p.type = "Wordvec"
        add_weight_filler(self.p.wordvec_param.weight_filler, 0.1)
        self.p.wordvec_param.dimension = dimension
        self.p.wordvec_param.vocab_size = vocab_size
        add_weight_filler(self.p.wordvec_param.weight_filler, init_range)

class ReluLayer(Layer):
    def __init__(self, **kwargs):
        super(ReluLayer, self).__init__(kwargs)
        self.p.type = "ReLU"

class DropoutLayer(Layer):
    def __init__(self, dropout_ratio, **kwargs):
        super(DropoutLayer, self).__init__(kwargs)
        self.p.type = "Dropout"
        self.p.dropout_param.dropout_ratio = dropout_ratio

class SoftmaxLayer(Layer):
    def __init__(self, ignore_label=None, normalize=None, **kwargs):
        super(SoftmaxLayer, self).__init__(kwargs)
        self.p.type = "Softmax"

class LossLayer(Layer):
    def __init__(self, kwargs):
        super(LossLayer ,self).__init__(kwargs)
        tops = kwargs.get('tops', [kwargs['name']])
        loss_weight = kwargs.get('loss_weight', 1.)
        self.p.loss_weight.append(loss_weight)
        assert len(tops) == 1

class EuclideanLossLayer(LossLayer):
    def __init__(self, **kwargs):
        super(EuclideanLossLayer, self).__init__(kwargs)
        self.p.type = "EuclideanLoss"

class SoftmaxWithLossLayer(LossLayer):
    def __init__(self, ignore_label=None, normalize=None, **kwargs):
        super(SoftmaxWithLossLayer, self).__init__(kwargs)
        self.p.type = "SoftmaxWithLoss"
        if normalize is not None:
            self.p.loss_param.normalize = normalize
        if ignore_label is not None:
            self.p.loss_param.ignore_label = ignore_label
