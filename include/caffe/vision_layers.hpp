#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/relu_layer.hpp"

namespace caffe {

// Forward declare PoolingLayer and SplitLayer for use in LRNLayer.
template <typename Dtype> class PoolingLayer;
template <typename Dtype> class SplitLayer;

}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_
