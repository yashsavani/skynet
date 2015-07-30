import caffe_pb2
import log
import utils

from ._apollo import Tensor, Net, Caffe, make_numpy_data_param, Blob

log.setup_logging()

set_mode_gpu = Caffe.set_mode_gpu
set_mode_cpu = Caffe.set_mode_cpu
set_logging_verbosity = Caffe.set_logging_verbosity
set_device = Caffe.set_device
set_logging_verbosity(3)

def set_random_seed(value):
    import numpy as np
    import random
    np.random.seed(value)
    random.seed(value)
    Caffe.set_random_seed(value)
