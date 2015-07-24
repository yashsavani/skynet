#ifndef CAFFE_WORDVEC_LAYER_HPP_
#define CAFFE_WORDVEC_LAYER_HPP_

#include "caffe/neuron_layers.hpp"

namespace caffe {

template <typename Dtype>
class WordvecLayer : public Layer<Dtype> {
 public:
  explicit WordvecLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline bool overwrites_param_diffs() { return true; }
  virtual inline const char* type() const { return "Wordvec"; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_;  // batch size;
  int vocab_size_;
  int dimension_;
  int sentence_length_;
};

} // namespace caffe

#endif // CAFFE_WORDVEC_LAYER_HPP_
