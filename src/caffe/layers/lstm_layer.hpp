#ifndef CAFFE_LSTM_LAYER_HPP_
#define CAFFE_LSTM_LAYER_HPP_

#include "caffe/neuron_layers.hpp"

namespace caffe {

template <typename Dtype>
class LstmLayer : public Layer<Dtype> {
 public:
  explicit LstmLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline bool overwrites_param_diffs() { return true; }
  virtual inline const char* type() const { return "Lstm"; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;  // num memory cells;
  int num_;  // batch size;
  int input_data_size_;
  int M_;
  int N_;
  int K_;
  shared_ptr<Blob<Dtype> > input_gates_data_buffer_;
  shared_ptr<Blob<Dtype> > forget_gates_data_buffer_;
  shared_ptr<Blob<Dtype> > output_gates_data_buffer_;
  shared_ptr<Blob<Dtype> > input_values_data_buffer_;
  shared_ptr<Blob<Dtype> > gates_diff_buffer_;
  shared_ptr<Blob<Dtype> > next_state_tot_diff_buffer_;
  shared_ptr<Blob<Dtype> > dldg_buffer_;
};

} // namespace caffe

#endif // CAFFE_LSTM_LAYER_HPP_
