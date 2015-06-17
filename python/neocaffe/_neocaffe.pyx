cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.map cimport pair
from cython.operator cimport postincrement as postincrement
from cython.operator cimport dereference as dereference

import caffe_pb2
from caffe_pb2 import DataParameter
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
        void reset(T*)

cdef extern from "caffe/layer.hpp" namespace "caffe":
    cdef cppclass Layer[float]:
        float Forward(vector[Blob*]& bottom, vector[Blob*]& top)
        float Backward(vector[Blob*]& top, vector[bool]& propagate_down, vector[Blob*]& bottom)

cdef extern from "caffe/blob.hpp" namespace "caffe":
    cdef cppclass Blob[float]:
        Blob()
        Blob(vector[int]&)
        vector[int] shape()
        int count()
        float* mutable_cpu_data()
        float* mutable_cpu_diff()

cdef extern from "caffe/proto/caffe.pb.h" namespace "caffe":
    cdef cppclass LayerParameter:
        bool ParseFromString(string& data)
        string name()
        string& bottom(int)
        int bottom_size()

cdef extern from "caffe/layer_factory.hpp" namespace "caffe::LayerRegistry<float>":
    cdef shared_ptr[Layer] CreateLayer(LayerParameter& param)
        
cdef class Net:
    #cdef NeoNet* thisptr
    cdef map[string, shared_ptr[Layer]] layers_map
    cdef map[string, vector[string]] bottom_blobs_names
    cdef vector[shared_ptr[Layer]] layers
    cdef map[string, shared_ptr[Blob]] top_blobs
    cdef map[string, vector[shared_ptr[Blob]]] bottom_blobs;
    def __init__(self):
        pass
    def ForwardLayer(self, python_param):
        cdef shared_ptr[Layer] layer
        cdef LayerParameter layer_param
        cdef string layer_name = layer_param.name()
        cdef bool new_layer = self.layers_map.find(layer_name) == self.layers_map.end();
        layer_param.ParseFromString(python_param.SerializeToString())
        if (new_layer):
            layer = CreateLayer(layer_param)
            self.layers_map[layer_name] = layer;
        else:
            layer = self.layers_map[layer_name];

        self.layers.push_back(layer);
        cdef vector[Blob*] bottom_vec;
        cdef vector[Blob*] top_vec;
        cdef vector[bool] propagate_down;

        cdef vector[string] bottom_names
        (&bottom_names)[0]= self.bottom_blobs_names[layer_name];
        cdef bool reset_bottoms = False;
        cdef int bottom_id = 0
        cdef string blob_name
        for bottom_id in range(layer_param.bottom_size()):
            (&blob_name)[0] = layer_param.bottom(bottom_id)
            if (self.top_blobs.find(blob_name) != self.top_blobs.end()):
                raise ValueError("Could not find bottom: %s for layer: %s" % (blob_name, layer_name))
            if (bottom_id >= bottom_names.size() or bottom_names[bottom_id] != blob_name):
                reset_bottoms = True;
                break
        cdef shared_ptr[Blob] top_blob
        cdef shared_ptr[Blob] blob_pointer
        if reset_bottoms:
            self.bottom_blobs[layer_name].clear();
            for bottom_id in range(layer_param.bottom_size()):
                (&blob_name)[0] = layer_param.bottom(bottom_id)
                top_blob.reset(self.top_blobs[blob_name])
                blob_pointer.reset(new Blob(top_blob.shape()))
              #blob_pointer->ShareData(*top_blob);
              #if (!layer->overwrites_delta()) {
                  ## if layer accumulates delta rather than overwriting
                  #blob_pointer->ShareDiff(*top_blob);
              #bottom_blobs_[layer_name].push_back(blob_pointer);

class PyLayer(object):
    def __init__(self):
        self.param = caffe_pb2.LayerParameter()
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

net = Net()
net.ForwardLayer(DataLayer().param)

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
