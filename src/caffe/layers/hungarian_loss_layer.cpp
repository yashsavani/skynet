#ifdef USE_DLIB
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <dlib/optimization/max_cost_assignment.h>
#include <climits>

namespace caffe {

template <typename Dtype>
void HungarianLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  const vector<int> target_confidence_shape(4);
  top[1]->Reshape(bottom[2]->shape());
  match_ratio_ = this->layer_param_.hungarian_loss_param().match_ratio();
  if (this->layer_param_.loss_weight_size() == 1) {
    this->layer_param_.add_loss_weight(Dtype(0));
  }
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[1]->num(), bottom[2]->num());
}

template <typename Dtype>
void HungarianLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* box_pred_data = bottom[0]->cpu_data();
  const Dtype* boxes_data = bottom[1]->cpu_data();
  const Dtype* box_flags_data = bottom[2]->cpu_data();
  Dtype loss = 0.;
  assignments_.clear();
  num_gt_.clear();
  Dtype* top_confidences = top[1]->mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    const int offset = n * bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
    num_gt_.push_back(0);
    for (int i = 0; i < bottom[2]->height(); ++i) {
      Dtype box_score = (*(box_flags_data + bottom[2]->offset(n, 0, i)));
      CHECK_NEAR(static_cast<int>(box_score), box_score, 0.01);
      num_gt_[n] += static_cast<int>(box_score);
    }
    const int channels = bottom[0]->channels();
    CHECK_EQ(channels, 4);

    const int num_pred = bottom[0]->height();

    dlib::matrix<float> loss_mat(num_pred,num_pred);
    dlib::matrix<float> match_bonus(num_pred,num_pred);
    const int height = bottom[0]->height();
    for (int i = 0; i < num_pred; ++i) {
      for (int j = 0; j < num_pred; ++j) {
        loss_mat(i, j) = 0;
        match_bonus(i,j) = 0;
        if (j >= num_gt_[n]) { continue; }
        for (int c = 0; c < channels; ++c) {
          const Dtype pred_value = box_pred_data[offset + c * height + i];
          const Dtype label_value = boxes_data[offset + c * height + j];
          match_bonus(i,j) -= fabs(pred_value - label_value) / 1000.;
          CHECK_LT(match_bonus(i,j), 0.2);
          loss_mat(i,j) += fabs(pred_value - label_value);
        }
        match_bonus(i, j) -= i;
        const int c_x = 0;
        const int c_y = 1;
        const int c_w = 2;
        const int c_h = 3;

        const Dtype pred_x = box_pred_data[offset + c_x * num_pred + i];
        const Dtype pred_y = box_pred_data[offset + c_y * num_pred + i];

        const Dtype label_x = boxes_data[offset + c_x * num_pred + j];
        const Dtype label_y = boxes_data[offset + c_y * num_pred + j];
        const Dtype label_w = boxes_data[offset + c_w * num_pred + j];
        const Dtype label_h = boxes_data[offset + c_h * num_pred + j];

        float ratio;
        if (this->phase_ == TRAIN) {
          ratio = match_ratio_;
        } else {
          ratio = 1.0;
        }

        if (fabs(pred_x - label_x) / label_w > ratio || fabs(pred_y - label_y) / label_h > ratio) {
          match_bonus(i, j) -= 100;
        }
      }
    }

    double max_pair_bonus = 0;
    for (int i = 0; i < num_pred; ++i) {
      for (int j = 0; j < num_pred; ++j) {
        max_pair_bonus = std::max(max_pair_bonus, fabs(match_bonus(i,j)));
      }
    }
    dlib::matrix<int> int_bonus(num_pred,num_pred);
    for (int i = 0; i < num_pred; ++i) {
      for (int j = 0; j < num_pred; ++j) {
        int_bonus(i,j) = static_cast<int>(match_bonus(i,j) / max_pair_bonus * Dtype(INT_MAX) / 2.);
      }
    }

    std::vector<long> assignment;
    if (this->layer_param_.hungarian_loss_param().permute_matches()) {
      assignment = dlib::max_cost_assignment(int_bonus);
    } else {
      for (int i = 0; i < height; ++i) {
        assignment.push_back(i);
      }
    }
    for (int i = 0; i < num_pred; ++i) {
      loss += loss_mat(i, assignment[i]);
    }
    assignments_.push_back(assignment);

    if (n == 147) {
      std::stringstream stream;
      stream << "Assignment: ";
      for (int i = 0; i < assignment.size(); i++) {
        if (assignment[i] < num_gt_[n]) {
          stream << "o";
        } else {
          stream << '-';
        }
      }
      LOG(INFO) << stream.str();
    }
    for (int i = 0; i < num_pred; ++i) {
      top_confidences[n * num_pred + i] = assignment[i] < num_gt_[n] ? Dtype(1) : Dtype(0);
    }
  }
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void HungarianLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* box_pred_data = bottom[0]->cpu_data();
  const Dtype* boxes_data = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();

  Dtype* box_pred_diff = bottom[0]->mutable_cpu_diff();

  caffe_set(bottom[0]->count(), Dtype(0), box_pred_diff);

  for (int n = 0; n < bottom[0]->num(); ++n) {
    const int offset = n * bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    CHECK_EQ(channels, 4);
    for (int i = 0; i < assignments_[n].size(); ++i) {
      const int j = assignments_[n][i];
      if (j >= num_gt_[n]) { continue; }
      for (int c = 0; c < channels; ++c) {
        const Dtype pred_value = box_pred_data[offset + c * height + i];
        const Dtype label_value = boxes_data[offset + c * height + j];
        box_pred_diff[offset + c * height + i] = top_diff[0] * (pred_value > label_value ? Dtype(1.) : Dtype(-1.));
        //box_pred_diff[offset + c * height + i] = top_diff[0] * (pred_value - label_value) / 20.;
      }
    }
  }
}

INSTANTIATE_CLASS(HungarianLossLayer);
REGISTER_LAYER_CLASS(HungarianLoss);

}  // namespace caffe
#endif
