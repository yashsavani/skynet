#ifndef CAFFE_L1_LOSS_LAYER_HPP_
#define CAFFE_L1_LOSS_LAYER_HPP_

#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
class L1LossLayer : public LossLayer<Dtype> {
 public:
  explicit L1LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "L1Loss"; }

  /**
   * Unlike most loss layers, in the L1LossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> sign_;
};

} // namespace caffe

#endif // CAFFE_L1_LOSS_LAYER_HPP_

