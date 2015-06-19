cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.map cimport pair
from cython.operator cimport postincrement as postincrement
from cython.operator cimport dereference as dereference

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

cdef extern from "caffe/caffe.hpp" namespace "caffe::Caffe":
    void set_random_seed(unsigned int)
    enum Brew:
        CPU = 0
        GPU = 1
    void set_mode(Brew)

cdef class Caffe:
    def __cinit__(self):
        pass
    @staticmethod
    def set_random_seed(seed):
        set_random_seed(seed)
    @staticmethod
    def set_mode_cpu():
        set_mode(CPU)
    @staticmethod
    def set_mode_gpu():
        set_mode(GPU)

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

cdef extern from "caffe/neonet.hpp" namespace "caffe":
    cdef cppclass NeoNet[float]:
        NeoNet()
        void Init()
        void ForwardLayer(string, bool)
        void Backward()
        void Update(float lr, float momentum)
        map[string, shared_ptr[Blob]]& blobs()
        map[string, vector[shared_ptr[Blob]]] params_map()
        void set_phase_test()
        void set_phase_train()

cdef extern from "caffe/proto/caffe.pb.h" namespace "caffe":
    cdef cppclass LayerParameter:
        bool ParseFromString(string& data)
        string name()
        string& bottom(int)
        int bottom_size()

cdef extern from "caffe/layer_factory.hpp" namespace "caffe::LayerRegistry<float>":
    cdef shared_ptr[Layer] CreateLayer(LayerParameter& param)

cdef class PyBlob(object):
    cdef shared_ptr[Blob] thisptr
    def __cinit__(self):
        pass
    cdef void Init(self, shared_ptr[Blob] other):
        self.thisptr = other
    def shape(self):
        return self.thisptr.get().shape()
    def count(self):
        return self.thisptr.get().count()
    def diff(self):
        result = tonumpyarray(self.thisptr.get().mutable_cpu_diff(),
                    self.thisptr.get().count())
        pynp.reshape(result, self.shape())
        return result
    def data(self):
        result = tonumpyarray(self.thisptr.get().mutable_cpu_data(),
                    self.thisptr.get().count())
        pynp.reshape(result, self.shape())
        return result
        
cdef class Net:
    cdef NeoNet* thisptr
    def __cinit__(self):
        self.thisptr = new NeoNet()
    def __dealloc__(self):
        del self.thisptr
    def forward_layer(self, layer, reshape_only=False):
        self.thisptr.ForwardLayer(layer.p.SerializeToString(), reshape_only)
    def backward(self):
        self.thisptr.Backward()
    def update(self, lr, momentum=0.):
        self.thisptr.Update(lr, momentum)
    def set_phase_train(self):
        self.thisptr.set_phase_train()
    def set_phase_test(self):
        self.thisptr.set_phase_test()
    property params:
        def __get__(self):
            cdef map[string, vector[shared_ptr[Blob]]] param_map
            (&param_map)[0] = self.thisptr.params_map()

            params = {}
            cdef map[string, vector[shared_ptr[Blob]]].iterator it = param_map.begin()
            cdef map[string, vector[shared_ptr[Blob]]].iterator end = param_map.end()
            cdef string layer_name
            cdef vector[shared_ptr[Blob]] blob_vec
            cdef shared_ptr[Blob] blob_ptr
            while it != end:
                layer_name = dereference(it).first
                blob_vec = dereference(it).second
                params[layer_name] = []
                for i in range(blob_vec.size()):
                    blob_ptr = blob_vec[i]
                    new_blob = PyBlob()
                    new_blob.Init(blob_ptr)
                    params[layer_name].append(new_blob)
                postincrement(it)

            return params
    property blobs:
        def __get__(self):
            cdef map[string, shared_ptr[Blob]] blob_map
            (&blob_map)[0] = self.thisptr.blobs()

            blobs = {}
            cdef map[string, shared_ptr[Blob]].iterator it = blob_map.begin()
            cdef map[string, shared_ptr[Blob]].iterator end = blob_map.end()
            cdef string blob_name
            cdef shared_ptr[Blob] blob_ptr
            while it != end:
                blob_name = dereference(it).first
                blob_ptr = dereference(it).second
                new_blob = PyBlob()
                new_blob.Init(blob_ptr)
                blobs[blob_name] = new_blob
                postincrement(it)

            return blobs


#cdef NeoNet net

#cdef void foo(NeoNet& net):
    #net.ForwardLayer(DummyDataLayer().param.SerializeToString(), True)
    #net.ForwardLayer(ReluLayer().param.SerializeToString(), True)
    #cdef Blob* b = net.blobs()['image'].get()
    ##print b.shape()
    ##cdef map[string, shared_ptr[Blob]] blob_map
    ##(&blob_map)[0] = net.blobs()
    ##cdef void set_blob_map(map[string, shared_ptr[Blob]]& blob_map):
    ##set_blob_map(blob_map)

    ##cdef map[string, shared_ptr[Blob]].iterator it = blob_map.begin()
    ##cdef map[string, shared_ptr[Blob]].iterator end = blob_map.end()
    ##while it != end:
        ##print dereference(it).first
        ##postincrement(it)

    #result = (tonumpyarray(b.mutable_cpu_data(), b.count()).reshape(b.shape()))
    #net.Backward()

#for i in range(10):
    #foo(net)
