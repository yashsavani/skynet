cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.map cimport pair

cdef extern from "boost/shared_ptr.hpp" namespace "boost":
    cdef cppclass shared_ptr[T]:
        T* get()
        void reset(T*)

cdef extern from "caffe/proto/caffe.pb.h" namespace "caffe":
    cdef cppclass LayerParameter:
        bool ParseFromString(string& data)
        bool SerializeToString(string*)
        string name()
        string& bottom(int)
        int bottom_size()

cdef extern from "caffe/tensor.hpp" namespace "caffe":
    cdef cppclass Tensor[float]:
        Tensor()
        Tensor(vector[int]&)
        vector[int] shape()
        int count()
        float* mutable_cpu_mem() except +
        void Reshape(vector[int]& shape) except +
        void AddFrom(Tensor& other) except +
        void MulFrom(Tensor& other) except +
        void AddMulFrom(Tensor& other, float alpha) except +
        void SetValues(float value) except +
        void scale(float value) except +
        void CopyFrom(Tensor& other) except +

cdef extern from "caffe/blob.hpp" namespace "caffe":
    cdef cppclass Blob[float]:
        Blob()
        Blob(vector[int]&)
        vector[int] shape()
        int count()
        float* mutable_cpu_data()
        float* mutable_cpu_diff()
        shared_ptr[Tensor] data()
        shared_ptr[Tensor] diff()

cdef extern from "caffe/layer.hpp" namespace "caffe":
    cdef cppclass Layer[float]:
        float Forward(vector[Blob*]& bottom, vector[Blob*]& top)
        float Backward(vector[Blob*]& top, vector[bool]& propagate_down, vector[Blob*]& bottom)
        LayerParameter& layer_param()
        vector[shared_ptr[Blob]]& blobs()
        vector[shared_ptr[Blob]]& buffers()

cdef extern from "caffe/apollonet.hpp" namespace "caffe":
    cdef cppclass ApolloNet[float]:
        ApolloNet()
        float ForwardLayer(string layer_param_string, string runtime_param_string) except +
        void BackwardLayer(string layer_name)
        void Update(float lr, float momentum, float decay_rate, float clip_gradients)
        void UpdateParam(string param_name, float lr, float momentum, float decay_rate)
        void ResetForward()
        float DiffL2Norm()
        map[string, shared_ptr[Blob]]& blobs()
        map[string, shared_ptr[Layer]]& layers()
        map[string, shared_ptr[Blob]]& params()
        map[string, float]& param_decay_mults()
        map[string, float]& param_lr_mults()
        void set_phase_test()
        void set_phase_train()
        void CopyTrainedLayersFrom(string trained_filename)
        vector[string]& active_layer_names()
        set[string]& active_param_names()
