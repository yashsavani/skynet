from ._apollo import Tensor, Net, Caffe, make_numpy_data_param, Blob
import caffe_pb2
import log
import utils

log.setup_logging()
Caffe.set_logging_verbosity(3)
