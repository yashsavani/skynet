cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.map cimport pair
from cython.operator cimport postincrement as postincrement
from cython.operator cimport dereference as dereference

from caffe_pb2 import LayerParameter, DataParameter
import numpy as pynp

np.import_array()
cdef public api tonumpyarray(float* data, long long size) with gil:
    #if not (data and size >= 0): raise ValueError
    cdef np.npy_intp dims = size
    #NOTE: it doesn't take ownership of `data`. You must free `data` yourself
    return np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT, <void*>data)

cdef extern from "boost/shared_ptr.hpp" namespace "boost":
    cdef cppclass shared_ptr[T]:
        T* get()

cdef extern from "caffe/layer.hpp" namespace "caffe":
    cdef cppclass Layer[T]:
        T Forward(vector[Blob*]& bottom, vector[Blob*]& top)
        T Backward(vector[Blob*]& top, vector[bool]& propagate_down, vector[Blob*]& bottom)

cdef extern from "caffe/blob.hpp" namespace "caffe":
    cdef cppclass Blob[float]:
        vector[int] shape()
        int count()
        float* mutable_cpu_data()
        float* mutable_cpu_diff()

cdef cppclass NeoNet:
    void NeoNet() except +
    void Init():
        pass
    void ForwardLayer(layer_param):
        print layer_param

class PyLayer(object):
    def __init__(self):
        self.param = LayerParameter()
class DataLayer(PyLayer):
    def __init__(self):
        super().__init__()
        self.param.type = "Data"
        self.param.name = "data"
        self.param.top.append("image")
        self.param.data_param.source = 'examples/mnist/mnist_train_lmdb'
        self.param.data_param.backend = DataParameter.LMDB
        self.param.data_param.batch_size = 2

class ReluLayer(PyLayer):
    def __init__(self):
        super().__init__()
        self.param.type = "ReLU"
        self.param.name = "relu"
        self.param.bottom.append("image")
        self.param.top.append("relu")

cdef NeoNet net
net.ForwardLayer(DataLayer())

#cdef extern from "caffe/neonet.hpp" namespace "caffe":
    #cdef cppclass NeoNet[T]:
        #void Init()
        #void ForwardLayer(string)
        #map[string, shared_ptr[Blob]]& blobs()

#cdef NeoNet[float] net
##cdef string cpp_string = data_layer.SerializeToString()

#net.ForwardLayer(DataLayer().param.SerializeToString())
#net.ForwardLayer(ReluLayer().param.SerializeToString())
#cdef Blob* b = net.blobs()['image'].get()
#print b.shape()
#cdef map[string, shared_ptr[Blob]] blob_map
#cdef void set_blob_map(map[string, shared_ptr[Blob]]& blob_map):
    #(&blob_map)[0] = net.blobs()
#set_blob_map(blob_map)

#cdef map[string, shared_ptr[Blob]].iterator it = blob_map.begin()
#cdef map[string, shared_ptr[Blob]].iterator end = blob_map.end()
#while it != end:
    #print dereference(it).first
    #postincrement(it)

#result = (tonumpyarray(b.mutable_cpu_data(), b.count()).reshape(b.shape()))
##for x in net.blobs():
    ##print x.first
##print 'hello world'
