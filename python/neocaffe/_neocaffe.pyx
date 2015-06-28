cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.map cimport pair
from cython.operator cimport postincrement as postincrement
from cython.operator cimport dereference as dereference

import numpy as pynp
import h5py
import os
import caffe_pb2
import sys

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
    void SetDevice(int)
    void set_logging_verbosity(int level)

cdef class Caffe:
    def __cinit__(self):
        pass
    @staticmethod
    def set_random_seed(seed):
        set_random_seed(seed)
    @staticmethod
    def set_device(device_id):
        SetDevice(device_id)
    @staticmethod
    def set_mode_cpu():
        set_mode(CPU)
    @staticmethod
    def set_mode_gpu():
        set_mode(GPU)
    @staticmethod
    def set_logging_verbosity(level):
        set_logging_verbosity(level)

cdef extern from "caffe/proto/caffe.pb.h" namespace "caffe":
    cdef cppclass LayerParameter:
        bool ParseFromString(string& data)
        bool SerializeToString(string*)
        string name()
        string& bottom(int)
        int bottom_size()

cdef extern from "caffe/layer.hpp" namespace "caffe":
    cdef cppclass Layer[float]:
        float Forward(vector[Blob*]& bottom, vector[Blob*]& top)
        float Backward(vector[Blob*]& top, vector[bool]& propagate_down, vector[Blob*]& bottom)
        LayerParameter& layer_param()
        vector[shared_ptr[Blob]] blobs()

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
        float ForwardLayer(string layer_param_string, string runtime_param_string) except +
        void Backward()
        void Update(float lr, float momentum, float clip_gradients)
        void ResetForward()
        map[string, shared_ptr[Blob]]& blobs()
        map[string, shared_ptr[Layer]]& layers()
        map[string, shared_ptr[Blob]]& params()
        void set_phase_test()
        void set_phase_train()
        void CopyTrainedLayersFrom(string trained_filename)

cdef extern from "caffe/layer_factory.hpp" namespace "caffe::LayerRegistry<float>":
    cdef shared_ptr[Layer] CreateLayer(LayerParameter& param)

cdef class PyLayer(object):
    cdef shared_ptr[Layer] thisptr
    def __cinit__(self):
        pass
    cdef void Init(self, shared_ptr[Layer] other):
        self.thisptr = other
    property layer_param:
        def __get__(self):
            param = caffe_pb2.LayerParameter()
            cdef string s
            self.thisptr.get().layer_param().SerializeToString(&s)
            param.ParseFromString(s)
            return param
    property params:
        def __get__(self):
            params = []
            cdef vector[shared_ptr[Blob]] cparams
            (&cparams)[0] = self.thisptr.get().blobs()
            for i in range(cparams.size()):
                new_blob = PyBlob()
                new_blob.Init(cparams[i])
                params.append(new_blob)
            return params

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
        sh = self.shape()
        result.shape = sh if len(sh) > 0 else (1,)
        return result
    def data(self):
        result = tonumpyarray(self.thisptr.get().mutable_cpu_data(),
                    self.thisptr.get().count())
        sh = self.shape()
        result.shape = sh if len(sh) > 0 else (1,)
        return result
        
cdef class Net:
    cdef NeoNet* thisptr
    def __cinit__(self, phase='train'):
        self.thisptr = new NeoNet()
        if phase == 'train':
            self.thisptr.set_phase_train()
        elif phase == 'test':
            self.thisptr.set_phase_test()
        else:
            assert False, "phase must be one of ['train', 'test']"
    def __dealloc__(self):
        del self.thisptr
    def forward(self, arch):
        return arch.forward(self)
    def forward_layer(self, layer):
        return self.thisptr.ForwardLayer(layer.p.SerializeToString(), layer.r.SerializeToString())
    def backward(self):
        self.thisptr.Backward()
    def update(self, lr, momentum=0., clip_gradients=-1):
        self.thisptr.Update(lr, momentum, clip_gradients)
    def reset_forward(self):
        self.thisptr.ResetForward()
    property layers:
        def __get__(self):
            cdef map[string, shared_ptr[Layer]] layers_map
            (&layers_map)[0] = self.thisptr.layers()

            layers = {}
            cdef map[string, shared_ptr[Layer]].iterator it = layers_map.begin()
            cdef map[string, shared_ptr[Layer]].iterator end = layers_map.end()
            cdef string layer_name
            cdef shared_ptr[Layer] layer
            while it != end:
                layer_name = dereference(it).first
                layer = dereference(it).second
                py_layer = PyLayer()
                py_layer.Init(layer)
                layers[layer_name] = py_layer
                postincrement(it)

            return layers
    property params:
        def __get__(self):
            cdef map[string, shared_ptr[Blob]] param_map
            (&param_map)[0] = self.thisptr.params()

            blobs = {}
            cdef map[string, shared_ptr[Blob]].iterator it = param_map.begin()
            cdef map[string, shared_ptr[Blob]].iterator end = param_map.end()
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

    def save(self, filename):
        assert filename.endswith('.h5'), "saving only supports h5 files"
        with h5py.File(filename, 'w') as f:
            for name, value in self.params.items():
                f[name] = value.data()

    def load(self, filename):
        if len(self.params) == 0:
            sys.stderr.write('WARNING, loading into empty net.')
        _, extension = os.path.splitext(filename)
        if extension == '.h5':
            with h5py.File(filename, 'r') as f:
                params = self.params
                for name, stored_value in f.items():
                    if name in params:
                        value = params[name].data()
                        value[:] = stored_value
        elif extension == '.caffemodel':
            self.thisptr.CopyTrainedLayersFrom(filename)
        else:
            assert False, "Error, filename is neither h5 nor caffemodel: %s, %s" % (filename, extension)
    def copy_params_from(self, other):
        self_params = self.params
        if len(self_params) == 0:
            sys.stderr.write('WARNING, copying into empty net.')
        for name, value in other.params.items():
            if name in self_params:
                data = self_params[name].data()
                data[:] = pynp.copy(value.data())


cdef extern from "caffe/proto/caffe.pb.h" namespace "caffe":
    cdef cppclass NumpyDataParameter:
        void add_data(float data)
        void add_shape(unsigned int shape)
        string DebugString()
    cdef cppclass RuntimeParameter:
        NumpyDataParameter* mutable_numpy_data_param()
        NumpyDataParameter numpy_data_param()
        bool SerializeToString(string*)
        string DebugString()

class PyRuntimeParameter(object):
    def __init__(self, result):
        self.result = result
    def SerializeToString(self):
        return self.result
def make_numpy_data_param(numpy_array):
    assert numpy_array.dtype == pynp.float32
    cdef vector[int] v
    for x in numpy_array.shape:
        v.push_back(x)
    cdef string s = make_numpy_data_param_fast(pynp.ascontiguousarray(numpy_array.flatten()), v)
    return PyRuntimeParameter(str(s)) #s.encode('utf-8'))

cdef string make_numpy_data_param_fast(np.ndarray[np.float32_t, ndim=1] numpy_array, vector[int] v):
    cdef RuntimeParameter runtime_param
    cdef NumpyDataParameter* numpy_param
    numpy_param = runtime_param.mutable_numpy_data_param()
    cdef int length = len(numpy_array)
    cdef int i
    for i in range(v.size()):
        numpy_param[0].add_shape(v[i])
    for i in range(length):
        numpy_param[0].add_data(numpy_array[i])
    cdef string s
    runtime_param.SerializeToString(&s)
    return s
