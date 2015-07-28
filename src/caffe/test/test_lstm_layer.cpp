#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define NUM_CELLS 3
#define BATCH_SIZE 4
#define INPUT_DATA_SIZE 5

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class LstmLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LstmLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),
        blob_bottom2_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1601);
    blob_bottom_->Reshape(BATCH_SIZE, INPUT_DATA_SIZE, 1, 1);
    blob_bottom2_->Reshape(BATCH_SIZE, NUM_CELLS, 1, 1);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    GaussianFiller<Dtype> filler2(filler_param);
    filler.Fill(this->blob_bottom_);
    filler2.Fill(this->blob_bottom2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom2_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_.push_back(blob_top2_);
  }
  virtual ~LstmLayerTest() { delete blob_bottom_; delete blob_bottom2_; delete blob_top_; delete blob_top2_; }
  void ReferenceLstmForward(const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void LstmLayerTest<TypeParam>::ReferenceLstmForward(
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top) {
  typedef typename TypeParam::Dtype Dtype;
  blob_top->Reshape(blob_bottom.num(), blob_bottom.channels(),
      blob_bottom.height(), blob_bottom.width());
  Dtype* top_data = blob_top->mutable_cpu_data();
  LstmParameter lstm_param = layer_param.lstm_param();
}

TYPED_TEST_CASE(LstmLayerTest, TestDtypesAndDevices);

TYPED_TEST(LstmLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LstmParameter* lstm_param = layer_param.mutable_lstm_param();
  lstm_param->set_num_cells(NUM_CELLS);
  lstm_param->mutable_input_weight_filler()->set_type("xavier");
  lstm_param->mutable_input_gate_weight_filler()->set_type("xavier");
  lstm_param->mutable_forget_gate_weight_filler()->set_type("xavier");
  lstm_param->mutable_output_gate_weight_filler()->set_type("xavier");
  LstmLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), BATCH_SIZE);
  EXPECT_EQ(this->blob_top_->channels(), NUM_CELLS);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(LstmLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LstmParameter* lstm_param = layer_param.mutable_lstm_param();
  lstm_param->set_num_cells(NUM_CELLS);
  lstm_param->mutable_input_weight_filler()->set_type("xavier");
  lstm_param->mutable_input_gate_weight_filler()->set_type("xavier");
  lstm_param->mutable_forget_gate_weight_filler()->set_type("xavier");
  lstm_param->mutable_output_gate_weight_filler()->set_type("xavier");

  LstmLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  for (int i = 0; i < this->blob_top2_->count(); ++i) {
    this->blob_top2_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
