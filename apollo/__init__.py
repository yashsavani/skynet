from ._apollo import Tensor, Net, Caffe, make_numpy_data_param, Blob
import caffe_pb2
import log
from architecture import Architecture
from training import default_train, default_parser, validate_hyper

log.setup_logging()
Caffe.set_logging_verbosity(3)
