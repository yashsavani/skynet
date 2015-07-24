#ifndef CAFFE_CAP_SEQUENCE_LAYER_HPP_
#define CAFFE_CAP_SEQUENCE_LAYER_HPP_

#include "caffe/common_layers.hpp"

namespace caffe {

/* CapSequence Layer
 * Used to collect the final RNN step from a batch with inputs of different lengths */
template <typename Dtype>
class CapSequenceLayer : public Layer<Dtype> {
 public:
  explicit CapSequenceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CapSequence"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline bool overwrites_bottom_diffs() { return false; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

} // namespace caffe

#endif // CAFFE_CAP_SEQUENCE_LAYER_HPP_
