"""
Set of classes for building protobuf layer parameters from python
"""

import caffe_pb2
import apollo
from caffe_pb2 import DataParameter
import numpy as np

class Layer(object):
    def __init__(self, kwargs):
        name = kwargs['name']
        bottoms = kwargs.get('bottoms', [])
        tops = kwargs.get('tops', [name])
        param_names = kwargs.get('param_names', [])
        param_lr_mults = kwargs.get('param_lr_mults', [])
        param_decay_mults = kwargs.get('param_decay_mults', [])
        assert type(tops) != str and type(bottoms) != str and type(param_names) != str
        self.p = caffe_pb2.LayerParameter()
        self.r = caffe_pb2.RuntimeParameter()
        self.p.name = name
        for blob_name in tops:
            self.p.top.append(blob_name)
        for blob_name in bottoms:
            self.p.bottom.append(blob_name)
        for i in range(max(len(param_names), len(param_lr_mults), len(param_decay_mults))):
            param = self.p.param.add()
            if param_names:
                param.name = param_names[i]
            if param_lr_mults:
                param.lr_mult = param_lr_mults[i]
            if param_decay_mults:
                param.decay_mult = param_decay_mults[i]
        if 'phase' in kwargs:
            if kwargs['phase'] == 'TRAIN':
                self.p.phase = caffe_pb2.TRAIN
            elif kwargs['phase'] == 'TEST':
                self.p.phase = caffe_pb2.TEST
            else:
                raise ValueError('Unknown phase')
        self.deploy = self.kwargs.get('deploy', True)
        self.train = self.kwargs.get('train', True)

class PyLayer(Layer):
    def __init__(self, kwargs):
        super(PyLayer ,self).__init__(kwargs)
        self.kwargs = kwargs
        self.p.type = 'Py'
        if 'param_shapes' in kwargs:
            for shape in kwarg['param_shapes']:
                param_shape = self.p.py_param.param_shapes.add()
                for dimension in shape:
                    param_shape.dimension.append(dimension)
        if 'param_fillers' in kwargs:
            assert len(kwargs['param_shapes']) == len(kwargs['param_filler'])
            for filler in kwarg['param_fillers']:
                filler_param = self.p.py_param.param_fillers.add()
                filler_param.CopyFrom(filler.filler_param)
    def setup(self, bottom_vec, top_vec):
        pass
    def forward(self, bottom_vec, top_vec):
        pass
    def backward(self, bottom_vec, top_vec):
        pass

class LossLayer(Layer):
    def __init__(self, kwargs):
        kwargs['deploy'] = kwargs.get('deploy', False)
        super(LossLayer ,self).__init__(kwargs)
        tops = kwargs.get('tops', [kwargs['name']])
        loss_weight = kwargs.get('loss_weight', 1.)
        self.p.loss_weight.append(loss_weight)
        assert len(tops) == 1

class Filler(object):
    def __init__(self, **kwargs):
        self.filler_param = caffe_pb2.FillerParameter()
        if 'type' in kwargs:
            self.filler_param.type = kwargs['type']
        if 'value' in kwargs:
            self.filler_param.value = kwargs['value']
        if 'min' in kwargs:
            self.filler_param.min= kwargs['min']
        if 'max' in kwargs:
            self.filler_param.max = kwargs['max']
        if 'mean' in kwargs:
            self.filler_param.mean = kwargs['mean']
        if 'std' in kwargs:
            self.filler_param.std = kwargs['std']
        if 'sparse' in kwargs:
            self.filler_param.sparse = kwargs['sparse']

class Transform(object):
    def __init__(self, **kwargs):
        self.transform_param = caffe_pb2.TransformationParameter()
        if 'scale' in kwargs:
            self.transform_param.scale = kwargs['scale']
        if 'mirror' in kwargs:
            self.transform_param.mirror = kwargs['mirror']
        if 'crop_size' in kwargs:
            self.transform_param.crop_size = kwargs['crop_size']
        if 'mean_file' in kwargs:
            self.transform_param.mean_file = kwargs['mean_file']
        if 'force_color' in kwargs:
            self.transform_param.force_color = kwargs['force_color']
        if 'force_gray' in kwargs:
            self.transform_param.force_gray = kwargs['force_gray']
        if 'mean_value' in kwargs:
            for x in kwargs['mean_value']:
                self.transform_param.mean_value.append(x)

# -------------------------------------------------------------------------------
# Begin list of layers
# -------------------------------------------------------------------------------

class CapSequence(Layer):
    def __init__(self, sequence_lengths, **kwargs):
        super(CapSequence, self).__init__(kwargs)
        self.p.type = type(self).__name__
        for x in sequence_lengths:
            self.r.cap_sequence_param.sequence_lengths.append(x)

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
        if 'bias_term' in kwargs:
            self.p.convolution_param.bias_term = kwargs['bias_term']
        if 'kernel_size' in kwargs:
            self.p.convolution_param.kernel_size = kwargs['kernel_size']
        if 'kernel_h' in kwargs:
            self.p.convolution_param.kernel_h = kwargs['kernel_h']
        if 'kernel_w' in kwargs:
            self.p.convolution_param.kernel_w = kwargs['kernel_w']
        if 'pad' in kwargs:
            self.p.convolution_param.pad = kwargs['pad']
        if 'pad_h' in kwargs:
            self.p.convolution_param.pad_h = kwargs['pad_h']
        if 'pad_w' in kwargs:
            self.p.convolution_param.pad_w = kwargs['pad_w']
        if 'stride' in kwargs:
            self.p.convolution_param.stride = kwargs['stride']
        if 'stride_h' in kwargs:
            self.p.convolution_param.stride_h = kwargs['stride_h']
        if 'stride_w' in kwargs:
            self.p.convolution_param.stride_w = kwargs['stride_w']
        if 'group' in kwargs:
            self.p.convolution_param.group = kwargs['group']
        if 'weight_filler' in kwargs:
            self.p.convolution_param.weight_filler.CopyFrom(kwargs['weight_filler'].filler_param)
        if 'bias_filler' in kwargs:
            self.p.convolution_param.bias_filler.CopyFrom(kwargs['bias_filler'].filler_param)

class Data(Layer):
    def __init__(self, source, batch_size, **kwargs):
        super(Data, self).__init__(kwargs)
        self.p.type = type(self).__name__
        self.p.data_param.source = source
        self.p.data_param.backend = DataParameter.LMDB
        self.p.data_param.batch_size = batch_size
        if 'transform' in kwargs:
            self.p.transform_param.CopyFrom(kwargs['transform'].transform_param)

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

class Eltwise(Layer):
    def __init__(self, operation, **kwargs):
        super(Eltwise, self).__init__(kwargs)
        self.p.type = type(self).__name__
        if operation == 'MAX':
            self.p.eltwise_param.operation = caffe_pb2.EltwiseParameter.MAX
        elif operation == 'SUM':
            self.p.eltwise_param.operation = caffe_pb2.EltwiseParameter.SUM
        elif operation == 'PROD':
            self.p.eltwise_param.operation = caffe_pb2.EltwiseParameter.PROD
        else:
            raise ValueError('Unknown Eltwise operator')

class EuclideanLoss(LossLayer):
    def __init__(self, **kwargs):
        super(EuclideanLoss, self).__init__(kwargs)
        self.p.type = type(self).__name__

class HDF5Data(Layer):
    def __init__(self, source, batch_size, **kwargs):
        super(HDF5Data, self).__init__(kwargs)
        self.p.type = type(self).__name__
        self.p.hdf5_data_param.source = source
        self.p.hdf5_data_param.batch_size = batch_size
        if 'transform' in kwargs:
            self.p.transform_param.CopyFrom(kwargs['transform'].transform_param)
class ImageData(Layer):
    def __init__(self, source, batch_size, new_height=None, new_width=None, **kwargs):
        super(ImageData, self).__init__(kwargs)
        self.p.type = type(self).__name__
        self.p.image_data_param.source = source
        self.p.image_data_param.batch_size = batch_size
        if new_height is not None:
            self.p.image_data_param.new_height = new_height
        if new_width is not None:
            self.p.image_data_param.new_width = new_width
        if 'transform' in kwargs:
            self.p.transform_param.CopyFrom(kwargs['transform'].transform_param)


class InnerProduct(Layer):
    def __init__(self, num_output, bias_term=None, output_4d=None, **kwargs):
        super(InnerProduct, self).__init__(kwargs)
        self.p.type = type(self).__name__
        self.p.inner_product_param.num_output = num_output
        if 'weight_filler' in kwargs:
            self.p.inner_product_param.weight_filler.CopyFrom(kwargs['weight_filler'].filler_param)
        if 'bias_filler' in kwargs:
            self.p.inner_product_param.bias_filler.CopyFrom(kwargs['bias_filler'].filler_param)
        if bias_term is not None:
            self.p.inner_product_param.bias_term = bias_term
        if output_4d is not None:
            self.p.inner_product_param.output_4d = output_4d

class LRN(Layer):
    def __init__(self, local_size, alpha, beta, **kwargs):
        super(LRN, self).__init__(kwargs)
        self.p.type = type(self).__name__
        self.p.lrn_param.local_size = local_size
        self.p.lrn_param.alpha = alpha
        self.p.lrn_param.beta = beta

class Lstm(Layer):
    def __init__(self, num_cells, **kwargs):
        super(Lstm, self).__init__(kwargs)
        self.p.type = type(self).__name__
        self.p.lstm_param.num_cells = num_cells
        if 'weight_filler' in kwargs:
            self.p.lstm_param.input_weight_filler.CopyFrom(kwargs['weight_filler'].filler_param)
            self.p.lstm_param.input_gate_weight_filler.CopyFrom(kwargs['weight_filler'].filler_param)
            self.p.lstm_param.forget_gate_weight_filler.CopyFrom(kwargs['weight_filler'].filler_param)
            self.p.lstm_param.output_gate_weight_filler.CopyFrom(kwargs['weight_filler'].filler_param)

class L1Loss(LossLayer):
    def __init__(self, **kwargs):
        super(L1Loss, self).__init__(kwargs)
        self.p.type = type(self).__name__

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
        if 'kernel_size' in kwargs:
            self.p.pooling_param.kernel_size = kwargs['kernel_size']
        if 'kernel_h' in kwargs:
            self.p.pooling_param.kernel_h = kwargs['kernel_h']
        if 'kernel_w' in kwargs:
            self.p.pooling_param.kernel_w = kwargs['kernel_w']
        if 'pad' in kwargs:
            self.p.pooling_param.pad = kwargs['pad']
        if 'pad_h' in kwargs:
            self.p.pooling_param.pad_h = kwargs['pad_h']
        if 'pad_w' in kwargs:
            self.p.pooling_param.pad_w = kwargs['pad_w']
        if 'stride' in kwargs:
            self.p.pooling_param.stride = kwargs['stride']
        if 'stride_h' in kwargs:
            self.p.pooling_param.stride_h = kwargs['stride_h']
        if 'stride_w' in kwargs:
            self.p.pooling_param.stride_w = kwargs['stride_w']
        if 'global_pooling' in kwargs:
            self.p.pooling_param.global_pooling = kwargs['global_pooling']
        if 'pool' in kwargs:
            if kwargs['pool'] == 'MAX':
                self.p.pooling_param.PoolMethod = caffe_pb2.PoolingParameter.MAX
            elif kwargs['pool'] == 'AVG':
                self.p.pooling_param.PoolMethod = caffe_pb2.PoolingParameter.AVG
            elif kwargs['pool'] == 'STOCHASTIC':
                self.p.pooling_param.PoolMethod = caffe_pb2.PoolingParameter.STOCHASTIC
            else:
                raise ValueError('Unknown pooling method')

class Power(Layer):
    def __init__(self, **kwargs):
        super(Power, self).__init__(kwargs)
        self.p.type = type(self).__name__
        if 'power' in kwargs:
            self.p.power_param.power = kwargs['power']
        if 'scale' in kwargs:
            self.p.power_param.scale = kwargs['scale']
        if 'shift' in kwargs:
            self.p.power_param.shift = kwargs['shift']

class ReLU(Layer):
    def __init__(self, **kwargs):
        super(ReLU, self).__init__(kwargs)
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

class Transpose(Layer):
    def __init__(self, ignore_label=None, normalize=None, **kwargs):
        super(Transpose, self).__init__(kwargs)
        self.p.type = type(self).__name__

class Unknown(Layer):
    def __init__(self, p, r=None):
        self.p = p
        if r is None:
            self.r = caffe_pb2.RuntimeParameter()
        else:
            self.r = r

class Wordvec(Layer):
    def __init__(self, dimension, vocab_size, **kwargs):
        super(Wordvec, self).__init__(kwargs)
        self.p.type = type(self).__name__
        self.p.wordvec_param.dimension = dimension
        self.p.wordvec_param.vocab_size = vocab_size
        if 'weight_filler' in kwargs:
            self.p.wordvec_param.weight_filler.CopyFrom(kwargs['weight_filler'].filler_param)

# =======================================================
# Python Layers
# =======================================================
class SamplePythonLayer(PyLayer):
    def __init__(self, **kwargs):
        super(SamplePythonLayer, self).__init__(kwargs)
    def forward(self, bottom, top):
        print len(bottom)
        print bottom[0].data
        print 'hello'

class Double(PyLayer):
    def __init__(self, **kwargs):
        super(Double, self).__init__(kwargs)
    def forward(self, bottom, top):
        top[0].reshape(bottom[0].shape)
        top[0].data_tensor.copy_from(bottom[0].data_tensor)
        top[0].data_tensor *= 2
    def backward(self, top, bottom):
        bottom[0].diff[:] = top[0].diff * 2
