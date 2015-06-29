cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
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
        float* mutable_cpu_mem()
        void Reshape(vector[int]& shape)
        void AddFrom(Tensor& other)
        void MulFrom(Tensor& other)
        void SetValues(float value)
        void scale(float value)
        void CopyFrom(Tensor& other)

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
        vector[shared_ptr[Blob]] blobs()
