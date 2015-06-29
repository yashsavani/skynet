from ._apollo import Tensor, Net, Caffe, make_numpy_data_param
import caffe_pb2
import logging
from architecture import Architecture

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
